#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import soundfile as sf
except Exception as exc:
    raise SystemExit(
        "Missing dependency: soundfile. Install via `pip install soundfile`."
    ) from exc


DEFAULT_PREPROCESS_DIR = Path(__file__).resolve().parent / "preprocess_output"
DEFAULT_STT_DIR = Path(__file__).resolve().parent / "stt_output"


@dataclass
class SttConfig:
    model_name: str = "medium"
    device: str = "cuda"
    language: str = "ko"
    task: str = "transcribe"
    fp16: bool = True
    beam_size: Optional[int] = None
    temperature: float = 0.0
    best_of: Optional[int] = None
    condition_on_previous_text: bool = True
    initial_prompt: Optional[str] = None
    include_tokens: bool = False


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_latest_preprocess_dir(root: Path) -> Path:
    if not root.exists():
        raise SystemExit(f"Preprocess output root not found: {root}")
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise SystemExit(f"No preprocess folders found in {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_input_audio(preprocess_dir: Path, prefer_vad: bool) -> Path:
    if not preprocess_dir.exists():
        raise SystemExit(f"Preprocess folder not found: {preprocess_dir}")

    vad_candidates = list(preprocess_dir.glob("*_preprocessed_vad.wav"))
    pre_candidates = list(preprocess_dir.glob("*_preprocessed.wav"))

    # Default policy: prefer ORIGINAL timeline unless explicitly prefer_vad
    if prefer_vad and vad_candidates:
        return vad_candidates[0]
    if pre_candidates:
        return pre_candidates[0]
    if vad_candidates:
        return vad_candidates[0]

    raise SystemExit(f"No preprocessed wav found in {preprocess_dir}")


def infer_origin_stem(audio_path: Path) -> str:
    stem = audio_path.stem
    if stem.endswith("_preprocessed_vad"):
        return stem[: -len("_preprocessed_vad")]
    if stem.endswith("_preprocessed"):
        return stem[: -len("_preprocessed")]
    return stem


def get_duration_sec(audio_path: Path) -> float:
    with sf.SoundFile(str(audio_path)) as f:
        return f.frames / float(f.samplerate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Whisper STT (local GPU) and save transcripts."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help=(
            "Input wav file or preprocess folder. "
            "Default: latest folder in preprocess_output"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Output root directory (default: stt_output)",
    )

    # Prefer VAD is experimental; default False (keep original timeline)
    try:
        from argparse import BooleanOptionalAction  # py3.9+
        parser.add_argument(
            "--prefer-vad",
            action=BooleanOptionalAction,
            default=False,
            help="Use *_preprocessed_vad.wav when available (default: false).",
        )
    except Exception:
        parser.add_argument("--prefer-vad", action="store_true", help="Prefer VAD wav.")
        parser.add_argument("--no-prefer-vad", action="store_true", help="Disable VAD preference.")

    parser.add_argument("--model", type=str, default="medium")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--language", type=str, default="ko")
    parser.add_argument("--task", type=str, default="transcribe")

    try:
        parser.add_argument("--fp16", action=BooleanOptionalAction, default=True)
    except Exception:
        parser.add_argument("--fp16", action="store_true", default=True)
        parser.add_argument("--no-fp16", action="store_true")

    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--best-of", type=int, default=None)
    parser.add_argument("--no-condition-on-previous-text", action="store_true")
    parser.add_argument("--initial-prompt", type=str, default=None)
    parser.add_argument("--include-tokens", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Compatibility for py<3.9 flags
    if hasattr(args, "no_prefer_vad"):
        prefer_vad = bool(getattr(args, "prefer_vad", False)) and (not args.no_prefer_vad)
    else:
        prefer_vad = bool(getattr(args, "prefer_vad", False))

    if hasattr(args, "no_fp16"):
        fp16 = bool(getattr(args, "fp16", True)) and (not args.no_fp16)
    else:
        fp16 = bool(getattr(args, "fp16", True))

    condition_on_previous_text = not args.no_condition_on_previous_text

    input_arg = Path(args.input) if args.input else None
    if input_arg is None:
        preprocess_dir = find_latest_preprocess_dir(DEFAULT_PREPROCESS_DIR)
        input_audio = resolve_input_audio(preprocess_dir, prefer_vad)
    elif input_arg.is_dir():
        input_audio = resolve_input_audio(input_arg, prefer_vad)
    else:
        input_audio = input_arg

    if not input_audio.exists():
        raise SystemExit(f"Input not found: {input_audio}")

    output_root = Path(args.output_root) if args.output_root else DEFAULT_STT_DIR
    origin_stem = infer_origin_stem(input_audio)
    output_dir = output_root / origin_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        import whisper
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: whisper/torch. "
            "Install via `pip install -U openai-whisper torch`."
        ) from exc

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # fp16 off on cpu to avoid errors
    if device == "cpu":
        fp16 = False

    cfg = SttConfig(
        model_name=args.model,
        device=device,
        language=args.language,
        task=args.task,
        fp16=fp16,
        beam_size=args.beam_size,
        temperature=args.temperature,
        best_of=args.best_of,
        condition_on_previous_text=condition_on_previous_text,
        initial_prompt=args.initial_prompt,
        include_tokens=args.include_tokens,
    )

    model = whisper.load_model(cfg.model_name, device=cfg.device)
    transcribe_kwargs: Dict[str, Any] = {
        "language": cfg.language,
        "task": cfg.task,
        "fp16": cfg.fp16,
        "condition_on_previous_text": cfg.condition_on_previous_text,
        "temperature": cfg.temperature,
    }
    if cfg.beam_size is not None:
        transcribe_kwargs["beam_size"] = cfg.beam_size
    if cfg.best_of is not None:
        transcribe_kwargs["best_of"] = cfg.best_of
    if cfg.initial_prompt:
        transcribe_kwargs["initial_prompt"] = cfg.initial_prompt

    result = model.transcribe(str(input_audio), **transcribe_kwargs)

    segments: List[Dict[str, Any]] = []
    for seg in result.get("segments", []):
        item = {
            "id": seg.get("id"),
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": (seg.get("text") or "").strip(),
            "avg_logprob": seg.get("avg_logprob"),
            "no_speech_prob": seg.get("no_speech_prob"),
            "compression_ratio": seg.get("compression_ratio"),
            "temperature": seg.get("temperature"),
        }
        if cfg.include_tokens and "tokens" in seg:
            item["tokens"] = seg["tokens"]
        segments.append(item)

    avg_logprob_vals = [s["avg_logprob"] for s in segments if s.get("avg_logprob") is not None]
    no_speech_vals = [s["no_speech_prob"] for s in segments if s.get("no_speech_prob") is not None]
    avg_logprob_mean = float(np.mean(avg_logprob_vals)) if avg_logprob_vals else None
    no_speech_prob_mean = float(np.mean(no_speech_vals)) if no_speech_vals else None

    transcript_text = (result.get("text") or "").strip()
    duration_sec = get_duration_sec(input_audio)
    vad_applied = input_audio.name.endswith("_preprocessed_vad.wav")

    created_at = utc_now_iso()
    transcript_payload = {
        "created_at": created_at,
        "model_name": cfg.model_name,
        "language": result.get("language", cfg.language),
        "duration_sec": duration_sec,
        "source_audio": str(input_audio),
        "vad_applied": vad_applied,
        "device": cfg.device,
        "text": transcript_text,
        "segments": segments,
        "stats": {
            "avg_logprob_mean": avg_logprob_mean,
            "no_speech_prob_mean": no_speech_prob_mean,
        },
    }

    transcript_path = output_dir / f"{origin_stem}_transcript.json"
    transcript_path.write_text(json.dumps(transcript_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    txt_path = output_dir / f"{origin_stem}_transcript.txt"
    txt_path.write_text(transcript_text, encoding="utf-8")

    meta_path = output_dir / "meta.json"
    meta_payload = {
        "created_at": created_at,
        "model_name": cfg.model_name,
        "device": cfg.device,
        "language": result.get("language", cfg.language),
        "task": cfg.task,
        "source_audio": str(input_audio),
        "vad_applied": vad_applied,
        "duration_sec": duration_sec,
        "avg_logprob_mean": avg_logprob_mean,
        "no_speech_prob_mean": no_speech_prob_mean,
        "beam_size": cfg.beam_size,
        "temperature": cfg.temperature,
        "best_of": cfg.best_of,
        "condition_on_previous_text": cfg.condition_on_previous_text,
        "initial_prompt": cfg.initial_prompt,
        "prefer_vad": prefer_vad,
    }
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {txt_path}")
    print(f"Saved: {transcript_path}")
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    main()
