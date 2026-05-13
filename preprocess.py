#!/usr/bin/env python3
import argparse
import io
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import soundfile as sf
except Exception as exc:
    raise SystemExit(
        "Missing dependency: soundfile. Install via `pip install soundfile`."
    ) from exc

try:
    from scipy.signal import resample_poly
except Exception as exc:
    raise SystemExit(
        "Missing dependency: scipy. Install via `pip install scipy`."
    ) from exc


TARGET_SR = 16000
DEFAULT_UPLOAD_DIR = Path(__file__).resolve().parent / "audio_uploads"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "preprocess_output"
DEFAULT_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}


@dataclass
class VadConfig:
    threshold: float = 0.5
    min_speech_sec: float = 0.3
    min_silence_sec: float = 0.2
    speech_pad_sec: float = 0.1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    """
    Try soundfile first; fallback to ffmpeg pipe decode for formats soundfile can't read.
    """
    try:
        audio, sr = sf.read(str(path), always_2d=True)
        mono = audio.mean(axis=1).astype(np.float32)
        return mono, sr
    except Exception:
        return load_audio_with_ffmpeg(path)


def load_audio_with_ffmpeg(path: Path) -> Tuple[np.ndarray, int]:
    ffmpeg_cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        "-f",
        "wav",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            ffmpeg_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "ffmpeg not found. Install ffmpeg to decode non-wav formats."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"ffmpeg failed to decode audio: {exc.stderr.decode(errors='ignore')}"
        ) from exc

    wav_bytes = io.BytesIO(result.stdout)
    audio, sr = sf.read(wav_bytes, always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    return mono, sr


def resample_audio(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    g = math.gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g
    return resample_poly(audio, up, down).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


def split_audio(audio: np.ndarray, sr: int, max_segment_sec: float) -> List[dict]:
    """
    Split audio into fixed-length chunks (seconds).
    Returns segment dicts with:
      - id
      - start/end (seconds)
      - start_sample/end_sample
    """
    segments: List[dict] = []
    if max_segment_sec <= 0:
        return segments

    max_samples = int(max_segment_sec * sr)
    total_samples = len(audio)

    start = 0
    seg_id = 0
    while start < total_samples:
        end = min(start + max_samples, total_samples)
        segments.append(
            {
                "id": seg_id,
                "start": start / sr,
                "end": end / sr,
                "start_sample": start,
                "end_sample": end,
            }
        )
        start = end
        seg_id += 1
    return segments


def save_split_segments(
    segments: List[dict],
    audio: np.ndarray,
    sr: int,
    output_dir: Path,
    source_path: Path,
    preprocessed_filename: str,
    write_audio: bool = True,
    status: str = "ok",
    error: Optional[str] = None,
) -> None:
    """
    Save split segments metadata + (optional) segment wav files.
    """
    if write_audio and segments:
        segments_dir = output_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        for seg in segments:
            start = int(seg["start_sample"])
            end = int(seg["end_sample"])
            seg_audio = audio[start:end]
            seg_path = segments_dir / f"segment_{seg['id']:04d}.wav"
            write_wav(seg_path, seg_audio, sr)
            seg["file"] = str(seg_path.name)
    else:
        for seg in segments:
            seg.setdefault("file", preprocessed_filename)

    meta_path = output_dir / f"{source_path.stem}_split_segments.json"
    meta = {
        "created_at": utc_now_iso(),
        "source": str(source_path),
        "sampling_rate": sr,
        "segment_count": len(segments),
        "status": status,
        "error": error,
        "segments": segments,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_silero_vad():
    try:
        import torch
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: torch. Install via `pip install torch`."
        ) from exc

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    get_speech_timestamps = utils[0]
    collect_chunks = utils[4]
    return model, get_speech_timestamps, collect_chunks


def run_vad(audio: np.ndarray, sr: int, vad_cfg: VadConfig) -> Tuple[List[dict], Optional[np.ndarray]]:
    """
    Returns:
      - segments: list of {id,start,end,start_sample,end_sample,text:""}
      - vad_audio: concatenated speech-only audio (or None)
    """
    model, get_speech_timestamps, collect_chunks = load_silero_vad()
    import torch

    audio_tensor = torch.from_numpy(audio)
    timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=sr,
        threshold=vad_cfg.threshold,
        min_speech_duration_ms=int(vad_cfg.min_speech_sec * 1000),
        min_silence_duration_ms=int(vad_cfg.min_silence_sec * 1000),
        speech_pad_ms=int(vad_cfg.speech_pad_sec * 1000),
    )

    segments: List[dict] = []
    for idx, ts in enumerate(timestamps):
        start_sample = int(ts["start"])
        end_sample = int(ts["end"])
        segments.append(
            {
                "id": idx,
                "start": start_sample / sr,
                "end": end_sample / sr,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "text": "",
            }
        )

    vad_audio = None
    if timestamps:
        vad_audio = collect_chunks(timestamps, audio_tensor).numpy()

    return segments, vad_audio


def save_vad_segments(
    segments: List[dict],
    output_dir: Path,
    source_path: Path,
    sr: int,
    vad_cfg: VadConfig,
    status: str = "ok",
    error: Optional[str] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{source_path.stem}_vad_segments.json"
    payload = {
        "created_at": utc_now_iso(),
        "source": str(source_path),
        "sampling_rate": sr,
        "vad_config": {
            "threshold": vad_cfg.threshold,
            "min_speech_sec": vad_cfg.min_speech_sec,
            "min_silence_sec": vad_cfg.min_silence_sec,
            "speech_pad_sec": vad_cfg.speech_pad_sec,
        },
        "status": status,
        "error": error,
        "segments": segments,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def find_latest_audio(upload_dir: Path) -> Path:
    if not upload_dir.exists():
        raise SystemExit(
            f"Upload directory not found: {upload_dir}. "
            "Create it or pass an explicit input path."
        )

    candidates = [
        p
        for p in upload_dir.iterdir()
        if p.is_file() and p.suffix.lower() in DEFAULT_AUDIO_EXTS
    ]
    if not candidates:
        raise SystemExit(
            f"No audio files found in {upload_dir}. "
            "Upload a file or pass an explicit input path."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess audio: mono/16kHz wav, optional VAD/split."
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help="Input audio file path (default: latest file in audio_uploads)",
    )

    # ✅ 추가: stt_run.py에서 넘기는 origin을 받기 위함
    parser.add_argument(
        "--origin",
        type=str,
        default=None,
        help="Output folder name (optional). If set, outputs go to <output-dir>/<origin>/",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write outputs (default: preprocess_output)",
    )
    parser.add_argument(
        "--max-segment-sec",
        type=float,
        default=0.0,
        help="Split audio into fixed segments (sec). 0 = single full segment in metadata only.",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable Silero VAD entirely (default: enabled).",
    )
    parser.add_argument(
        "--no-apply-vad",
        action="store_true",
        help="Do not write *_preprocessed_vad.wav (speech-only concatenated audio).",
    )
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-min-speech-sec", type=float, default=0.3)
    parser.add_argument("--vad-min-silence-sec", type=float, default=0.2)
    parser.add_argument("--vad-speech-pad-sec", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input:
        input_path = Path(args.input)
    else:
        input_path = find_latest_audio(DEFAULT_UPLOAD_DIR)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_root = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR

    # ✅ 변경: origin이 오면 그걸로 폴더명 고정 (stt_run.py와 폴더명 일치)
    run_name = args.origin if args.origin else input_path.stem
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = load_audio(input_path)
    audio = resample_audio(audio, sr, TARGET_SR)
    sr = TARGET_SR

    preprocessed_path = output_dir / f"{input_path.stem}_preprocessed.wav"
    write_wav(preprocessed_path, audio, sr)

    # Split metadata (+ optional segment wavs)
    try:
        if args.max_segment_sec <= 0:
            split_segments = [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": len(audio) / sr,
                    "start_sample": 0,
                    "end_sample": len(audio),
                    "file": preprocessed_path.name,
                }
            ]
            save_split_segments(
                split_segments,
                audio,
                sr,
                output_dir,
                input_path,
                preprocessed_path.name,
                write_audio=False,
                status="ok",
                error=None,
            )
        else:
            split_segments = split_audio(audio, sr, args.max_segment_sec)
            save_split_segments(
                split_segments,
                audio,
                sr,
                output_dir,
                input_path,
                preprocessed_path.name,
                write_audio=True,
                status="ok",
                error=None,
            )
    except Exception as exc:
        save_split_segments(
            [],
            audio,
            sr,
            output_dir,
            input_path,
            preprocessed_path.name,
            write_audio=False,
            status="failed",
            error=str(exc),
        )

    # VAD (segments json + optional speech-only wav)
    if not args.no_vad:
        vad_cfg = VadConfig(
            threshold=args.vad_threshold,
            min_speech_sec=args.vad_min_speech_sec,
            min_silence_sec=args.vad_min_silence_sec,
            speech_pad_sec=args.vad_speech_pad_sec,
        )
        try:
            segments, vad_audio = run_vad(audio, sr, vad_cfg)
            save_vad_segments(
                segments,
                output_dir,
                input_path,
                sr,
                vad_cfg,
                status="ok",
                error=None,
            )

            if (not args.no_apply_vad) and (vad_audio is not None):
                write_wav(
                    output_dir / f"{input_path.stem}_preprocessed_vad.wav",
                    vad_audio,
                    sr,
                )
        except Exception as exc:
            save_vad_segments(
                [],
                output_dir,
                input_path,
                sr,
                vad_cfg,
                status="failed",
                error=str(exc),
            )

    print(f"Saved: {preprocessed_path}")


if __name__ == "__main__":
    main()
