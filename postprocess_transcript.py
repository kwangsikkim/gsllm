#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_STT_DIR = Path(__file__).resolve().parent / "stt_output"

FILLERS = [
    "어",
    "음",
    "그",
    "저",
    "어어",
    "음음",
    "그그",
    "저기",
    "에",
    "뭐지",
]

FILLER_PATTERN = re.compile(r"\b(" + "|".join(map(re.escape, FILLERS)) + r")+\b")
MULTISPACE_PATTERN = re.compile(r"[ \t]{2,}")
MULTINEWLINE_PATTERN = re.compile(r"\n{3,}")
SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.:;!?…~])")

UNITS = {
    "퍼센트": "%",
    "프로": "%",
    "퍼센": "%",
    "킬로그램": "kg",
    "그램": "g",
    "킬로": "km",
    "미터": "m",
    "센티": "cm",
    "원": "원",
    "엔": "엔",
    "달러": "달러",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_units(text: str) -> str:
    for src, dst in UNITS.items():
        pattern = re.compile(rf"(\d+)\s*{re.escape(src)}\b")
        text = pattern.sub(lambda m, _dst=dst: f"{m.group(1)}{_dst}", text)
    return text


def normalize_basic(
    text: str,
    *,
    remove_fillers: bool = True,
    min_len_keep_fillers: int = 6,
) -> str:
    text = text.replace("\u00a0", " ")

    if remove_fillers and len(text.strip()) >= min_len_keep_fillers:
        text = FILLER_PATTERN.sub("", text)

    text = normalize_units(text)
    text = MULTISPACE_PATTERN.sub(" ", text).strip()
    text = SPACE_BEFORE_PUNCT.sub(r"\1", text)

    text = re.sub(r"[.]{3,}", "…", text)
    text = re.sub(r"[!]{2,}", "!", text)
    text = re.sub(r"[?]{2,}", "?", text)
    return text


def normalize_linebreaks(text: str) -> str:
    return MULTINEWLINE_PATTERN.sub("\n\n", text).strip()


def clean_segment_text(
    text: str,
    *,
    remove_fillers: bool = True,
    min_len_keep_fillers: int = 6,
) -> str:
    return normalize_basic(
        text,
        remove_fillers=remove_fillers,
        min_len_keep_fillers=min_len_keep_fillers,
    )


def _get_seg_times(seg: Dict) -> Tuple[Optional[float], Optional[float]]:
    start = seg.get("start")
    end = seg.get("end")
    try:
        start = float(start) if start is not None else None
    except Exception:
        start = None
    try:
        end = float(end) if end is not None else None
    except Exception:
        end = None
    return start, end


def merge_segments_by_speaker(
    segments: List[Dict],
    *,
    merge_same_speaker: bool = True,
    merge_gap_sec: float = 1.2,
    max_chars_per_block: int = 500,
) -> List[Dict]:
    if not merge_same_speaker or not segments:
        return segments

    merged: List[Dict] = []
    cur: Optional[Dict] = None

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        speaker = seg.get("speaker")
        start, end = _get_seg_times(seg)

        if cur is None:
            cur = dict(seg)
            cur["text"] = text
            merged.append(cur)
            continue

        cur_speaker = cur.get("speaker")
        cur_start, cur_end = _get_seg_times(cur)

        can_merge_speaker = (speaker == cur_speaker)
        can_merge_gap = True

        if cur_end is not None and start is not None:
            gap = start - cur_end
            can_merge_gap = gap <= merge_gap_sec

        if can_merge_speaker and can_merge_gap and (
            len(cur.get("text", "")) + 1 + len(text) <= max_chars_per_block
        ):
            cur["text"] = (cur.get("text", "") + " " + text).strip()
            if cur_start is None and start is not None:
                cur["start"] = start
            if end is not None:
                cur["end"] = end
        else:
            cur = dict(seg)
            cur["text"] = text
            merged.append(cur)

    return merged


def merge_segments_text(segments: List[Dict]) -> str:
    lines: List[str] = []
    for seg in segments:
        speaker = seg.get("speaker")
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if speaker:
            lines.append(f"{speaker}: {text}")
        else:
            lines.append(text)
    return normalize_linebreaks("\n".join(lines))


def find_latest_stt_dir(root: Path) -> Path:
    if not root.exists():
        raise SystemExit(f"STT output root not found: {root}")
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise SystemExit(f"No STT folders found in {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _pick_latest(files: List[Path]) -> Optional[Path]:
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def load_transcript_json(stt_dir: Path, origin: Optional[str]) -> Path:
    if origin:
        target = stt_dir / origin / f"{origin}_diarized_transcript.json"
        if target.exists():
            return target
        target = stt_dir / origin / f"{origin}_transcript.json"
        if target.exists():
            return target
        raise SystemExit(f"No transcript json found in {stt_dir / origin}")

    latest = find_latest_stt_dir(stt_dir)
    diarized = _pick_latest(list(latest.glob("*_diarized_transcript.json")))
    if diarized:
        return diarized
    regular = _pick_latest(list(latest.glob("*_transcript.json")))
    if regular:
        return regular
    raise SystemExit(f"No transcript json found in {latest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean transcript text for summarization.")
    parser.add_argument(
        "origin",
        nargs="?",
        default=None,
        help="Origin file stem (default: latest stt_output folder).",
    )
    parser.add_argument(
        "--stt-root",
        type=str,
        default=None,
        help="STT output root (default: stt_output)",
    )

    parser.add_argument("--keep-fillers", action="store_true", help="Do not remove filler words.")
    parser.add_argument(
        "--min-len-keep-fillers",
        type=int,
        default=6,
        help="If text length is shorter than this, keep fillers even when removing fillers.",
    )
    parser.add_argument("--drop-empty", action="store_true", help="Drop segments that become empty.")

    parser.add_argument("--no-merge-speaker", action="store_true", help="Do not merge same-speaker segments.")
    parser.add_argument("--merge-gap", type=float, default=1.2)
    parser.add_argument("--max-chars-per-block", type=int, default=500)
    return parser.parse_args()


def _origin_stem_from_filename(name: str) -> str:
    if name.endswith("_diarized_transcript.json"):
        return name[: -len("_diarized_transcript.json")]
    if name.endswith("_transcript.json"):
        return name[: -len("_transcript.json")]
    return Path(name).stem


def main() -> None:
    args = parse_args()
    stt_root = Path(args.stt_root) if args.stt_root else DEFAULT_STT_DIR
    transcript_path = load_transcript_json(stt_root, args.origin)

    try:
        raw = transcript_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as exc:
        raise SystemExit(f"Failed to read/parse JSON: {transcript_path}\n{exc}") from exc

    segments = data.get("segments", [])
    cleaned_segments: List[Dict] = []

    remove_fillers = not args.keep_fillers

    for seg in segments:
        text = seg.get("text", "")
        cleaned = clean_segment_text(
            text,
            remove_fillers=remove_fillers,
            min_len_keep_fillers=args.min_len_keep_fillers,
        )

        if args.drop_empty and not cleaned.strip():
            continue

        new_seg = dict(seg)
        new_seg["text"] = cleaned
        cleaned_segments.append(new_seg)

    merged_segments = merge_segments_by_speaker(
        cleaned_segments,
        merge_same_speaker=not args.no_merge_speaker,
        merge_gap_sec=args.merge_gap,
        max_chars_per_block=args.max_chars_per_block,
    )

    cleaned_text = merge_segments_text(merged_segments)
    created_at = utc_now_iso()

    output_dir = transcript_path.parent
    origin_stem = _origin_stem_from_filename(transcript_path.name)

    clean_json_path = output_dir / f"{origin_stem}_clean_transcript.json"
    clean_txt_path = output_dir / f"{origin_stem}_clean_transcript.txt"

    payload = dict(data)
    payload["created_at"] = created_at
    payload["segments"] = merged_segments
    payload["cleaned"] = True
    payload["postprocess"] = {
        "remove_fillers": remove_fillers,
        "min_len_keep_fillers": args.min_len_keep_fillers,
        "drop_empty": bool(args.drop_empty),
        "merge_same_speaker": not args.no_merge_speaker,
        "merge_gap_sec": args.merge_gap,
        "max_chars_per_block": args.max_chars_per_block,
    }

    clean_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    clean_txt_path.write_text(cleaned_text, encoding="utf-8")

    print(f"Loaded: {transcript_path}")
    print(f"Saved:  {clean_json_path}")
    print(f"Saved:  {clean_txt_path}")


if __name__ == "__main__":
    main()
