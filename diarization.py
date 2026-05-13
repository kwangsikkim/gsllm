#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_PREPROCESS_DIR = Path(__file__).resolve().parent / "preprocess_output"
DEFAULT_STT_DIR = Path(__file__).resolve().parent / "stt_output"


@dataclass
class DiarizeConfig:
    model_id: str = "pyannote/speaker-diarization-3.1"
    device: str = "cuda"
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_latest_dir(root: Path) -> Path:
    if not root.exists():
        raise SystemExit(f"Directory not found: {root}")
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise SystemExit(f"No folders found in {root}")
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


def load_stt_transcript(stt_root: Path, origin_stem: str) -> Optional[dict]:
    transcript_path = stt_root / origin_stem / f"{origin_stem}_transcript.json"
    if not transcript_path.exists():
        return None
    return json.loads(transcript_path.read_text(encoding="utf-8"))


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0.0, end - start)


def assign_speakers_to_segments(
    stt_segments: List[Dict[str, Any]],
    diar_segments: List[Dict[str, Any]],
    *,
    min_overlap_ratio: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Assign speaker by maximum overlap.
    If min_overlap_ratio > 0, require overlap / stt_seg_len >= ratio else UNKNOWN.
    """
    labeled: List[Dict[str, Any]] = []
    for seg in stt_segments:
        s_start = float(seg.get("start", 0.0) or 0.0)
        s_end = float(seg.get("end", 0.0) or 0.0)
        seg_len = max(1e-6, s_end - s_start)

        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for dseg in diar_segments:
            d_start = float(dseg["start"])
            d_end = float(dseg["end"])
            ov = overlap_seconds(s_start, s_end, d_start, d_end)
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = dseg["speaker"]

        if min_overlap_ratio > 0 and (best_overlap / seg_len) < min_overlap_ratio:
            best_speaker = "UNKNOWN"

        labeled_seg = dict(seg)
        labeled_seg["speaker"] = best_speaker
        labeled.append(labeled_seg)

    return labeled


def resolve_annotation(diarization: Any) -> Any:
    if hasattr(diarization, "itertracks"):
        return diarization
    if hasattr(diarization, "speaker_diarization"):
        ann = getattr(diarization, "speaker_diarization", None)
        if ann is not None and hasattr(ann, "itertracks"):
            return ann
    if hasattr(diarization, "exclusive_speaker_diarization"):
        ann = getattr(diarization, "exclusive_speaker_diarization", None)
        if ann is not None and hasattr(ann, "itertracks"):
            return ann
    for attr in ("annotation", "diarization", "_annotation"):
        value = getattr(diarization, attr, None)
        if value is not None and hasattr(value, "itertracks"):
            return value
        if isinstance(value, dict) and "annotation" in value:
            ann = value["annotation"]
            if hasattr(ann, "itertracks"):
                return ann
    if isinstance(diarization, dict) and "annotation" in diarization:
        ann = diarization["annotation"]
        if hasattr(ann, "itertracks"):
            return ann
    if hasattr(diarization, "to_annotation"):
        ann = diarization.to_annotation()
        if hasattr(ann, "itertracks"):
            return ann
    raise SystemExit(
        "Unsupported diarization output format. "
        "Please report the pyannote version and output type."
    )


def build_diarized_text(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Speaker diarization using pyannote/speaker-diarization-3.1"
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
        help="STT output root directory (default: stt_output)",
    )

    # Prefer VAD is experimental; default False (keep original timeline)
    try:
        from argparse import BooleanOptionalAction
        parser.add_argument(
            "--prefer-vad",
            action=BooleanOptionalAction,
            default=False,
            help="Use *_preprocessed_vad.wav when available (default: false).",
        )
    except Exception:
        parser.add_argument("--prefer-vad", action="store_true", help="Prefer VAD wav.")
        parser.add_argument("--no-prefer-vad", action="store_true", help="Disable VAD preference.")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=0.0,
        help="Require overlap/segment_length >= ratio else UNKNOWN (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if hasattr(args, "no_prefer_vad"):
        prefer_vad = bool(getattr(args, "prefer_vad", False)) and (not args.no_prefer_vad)
    else:
        prefer_vad = bool(getattr(args, "prefer_vad", False))

    input_arg = Path(args.input) if args.input else None
    if input_arg is None:
        preprocess_dir = find_latest_dir(DEFAULT_PREPROCESS_DIR)
        input_audio = resolve_input_audio(preprocess_dir, prefer_vad)
    elif input_arg.is_dir():
        input_audio = resolve_input_audio(input_arg, prefer_vad)
    else:
        input_audio = input_arg

    if not input_audio.exists():
        raise SystemExit(f"Input not found: {input_audio}")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit(
            "Hugging Face token not found. Set HF_TOKEN or HUGGINGFACE_TOKEN."
        )

    try:
        import torch
        from pyannote.audio import Pipeline
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: pyannote.audio/torch. "
            "Install via `pip install pyannote.audio torch`."
        ) from exc

    cfg = DiarizeConfig(
        device=args.device,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    try:
        pipeline = Pipeline.from_pretrained(cfg.model_id, use_auth_token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(cfg.model_id, token=token)

    if cfg.device == "cuda" and torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    else:
        pipeline.to(torch.device("cpu"))
        cfg.device = "cpu"

    diarization = pipeline(
        str(input_audio),
        num_speakers=cfg.num_speakers,
        min_speakers=cfg.min_speakers,
        max_speakers=cfg.max_speakers,
    )

    annotation = resolve_annotation(diarization)

    diar_segments: List[Dict[str, Any]] = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        diar_segments.append(
            {"start": float(segment.start), "end": float(segment.end), "speaker": speaker}
        )

    origin_stem = infer_origin_stem(input_audio)
    output_root = Path(args.output_root) if args.output_root else DEFAULT_STT_DIR
    output_dir = output_root / origin_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    created_at = utc_now_iso()
    diar_payload = {
        "created_at": created_at,
        "model_id": cfg.model_id,
        "device": cfg.device,
        "source_audio": str(input_audio),
        "prefer_vad": prefer_vad,
        "num_speakers": cfg.num_speakers,
        "min_speakers": cfg.min_speakers,
        "max_speakers": cfg.max_speakers,
        "segments": diar_segments,
    }
    diar_path = output_dir / f"{origin_stem}_diarization.json"
    diar_path.write_text(json.dumps(diar_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Speaker-label STT transcript (if exists)
    stt_data = load_stt_transcript(output_root, origin_stem)
    if stt_data and isinstance(stt_data.get("segments"), list):
        labeled_segments = assign_speakers_to_segments(
            stt_data["segments"],
            diar_segments,
            min_overlap_ratio=float(args.min_overlap_ratio or 0.0),
        )
        diarized_payload = dict(stt_data)
        diarized_payload["created_at"] = created_at
        diarized_payload["segments"] = labeled_segments
        diarized_payload["speaker_labeled"] = True

        diarized_path = output_dir / f"{origin_stem}_diarized_transcript.json"
        diarized_path.write_text(
            json.dumps(diarized_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        diarized_text = build_diarized_text(labeled_segments)
        diarized_txt_path = output_dir / f"{origin_stem}_diarized_transcript.txt"
        diarized_txt_path.write_text(diarized_text, encoding="utf-8")

        print(f"Saved: {diarized_path}")
        print(f"Saved: {diarized_txt_path}")

    print(f"Saved: {diar_path}")


if __name__ == "__main__":
    main()
