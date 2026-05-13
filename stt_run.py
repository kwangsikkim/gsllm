#!/usr/bin/env python3
import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except Exception as exc:
    raise SystemExit(
        "Missing dependency: numpy. Install via `pip install numpy`."
    ) from exc

try:
    import soundfile as sf
except Exception as exc:
    raise SystemExit(
        "Missing dependency: soundfile. Install via `pip install soundfile`."
    ) from exc

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_UPLOAD_DIR = BASE_DIR / "audio_uploads"
DEFAULT_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}

DEFAULT_PREPROCESS_DIR = BASE_DIR / "preprocess_output"
DEFAULT_STT_DIR = BASE_DIR / "stt_output"
DEFAULT_SUMMARY_DIR = BASE_DIR / "summary_output"
DEFAULT_REPORT_DIR = BASE_DIR / "report_output"
DEFAULT_DB_DIR = BASE_DIR / "chroma_db"

SCRIPT_PREPROCESS = BASE_DIR / "preprocess.py"
SCRIPT_STT = BASE_DIR / "stt.py"
SCRIPT_DIARIZATION = BASE_DIR / "diarization.py"
SCRIPT_POSTPROCESS = BASE_DIR / "postprocess_transcript.py"
SCRIPT_SUMMARY = BASE_DIR / "summary.py"
SCRIPT_QC = BASE_DIR / "quality_check.py"
SCRIPT_EXPORT = BASE_DIR / "export_report.py"
SCRIPT_EMBED = BASE_DIR / "report_embed.py"

GENV_PYTHON = BASE_DIR / "genv" / "bin" / "python"
PYTHON_BIN = str(GENV_PYTHON) if GENV_PYTHON.exists() else sys.executable


def _print_cmd(cmd: List[str]) -> None:
    quoted = " ".join(shlex.quote(c) for c in cmd)
    print(f"[stt_run] $ {quoted}")


def _run(cmd: List[str], *, log_path: Optional[Path] = None) -> None:
    _print_cmd(cmd)
    if not log_path:
        subprocess.run(cmd, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n[{utc_now_iso()}] $ {' '.join(cmd)}\n")
        logf.flush()
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)


def _append_arg(cmd: List[str], flag: str, value: Optional[object]) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _append_flag(cmd: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def _find_latest_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_latest_audio(upload_dir: Path) -> Optional[Path]:
    if not upload_dir.exists():
        return None
    candidates = [
        p for p in upload_dir.iterdir() if p.is_file() and p.suffix.lower() in DEFAULT_AUDIO_EXTS
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(status_path: Path, payload: Dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = status_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(status_path)


def _init_status(job_dir: Path, input_file: Optional[str]) -> Path:
    status_path = job_dir / "status.json"
    payload = {
        "state": "running",
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "input_file": input_file,
        "current_stage": None,
        "stages": [],
        "result_zip": None,
        "error": None,
    }
    _write_status(status_path, payload)
    return status_path


def _update_status_stage(status_path: Path, stage_name: str) -> None:
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    payload["updated_at"] = utc_now_iso()
    payload["current_stage"] = stage_name
    _write_status(status_path, payload)


def _append_status_stage(status_path: Path, stage_name: str, cmd: List[str], returncode: int) -> None:
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    payload["updated_at"] = utc_now_iso()
    payload["stages"].append(
        {
            "name": stage_name,
            "returncode": returncode,
            "command": cmd,
        }
    )
    _write_status(status_path, payload)


def _finish_status(status_path: Path, *, result_zip: Optional[str], error: Optional[str]) -> None:
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    payload["updated_at"] = utc_now_iso()
    payload["state"] = "done" if error is None else "failed"
    payload["current_stage"] = None
    payload["result_zip"] = result_zip
    payload["error"] = error
    _write_status(status_path, payload)


def _make_result_zip(job_dir: Path, zip_name: str = "result.zip") -> Path:
    zip_path = job_dir / zip_name
    if zip_path.exists():
        zip_path.unlink()
    base = str(zip_path).replace(".zip", "")
    shutil.make_archive(base, "zip", root_dir=str(job_dir))
    return zip_path


def _resolve_origin(
    origin_arg: Optional[str],
    input_path: Optional[str],
    preprocess_root: Path,
    stt_root: Path,
    summary_root: Path,
) -> Optional[str]:
    if origin_arg:
        return origin_arg
    if input_path:
        return Path(input_path).stem
    for root in (preprocess_root, stt_root, summary_root):
        latest = _find_latest_dir(root)
        if latest:
            return latest.name
    return None


def _make_display_title(input_path: Optional[str]) -> Optional[str]:
    if not input_path:
        return None
    stem = Path(input_path).stem
    date_str = datetime.now().strftime("%Y%m%d")
    return f"{stem}_{date_str}"


def _resolve_preprocessed_wav(preprocess_root: Path, origin: str, prefer_vad: bool) -> Optional[Path]:
    base_dir = preprocess_root / origin
    if not base_dir.exists():
        return None
    vad_path = base_dir / f"{origin}_preprocessed_vad.wav"
    pre_path = base_dir / f"{origin}_preprocessed.wav"
    if prefer_vad and vad_path.exists():
        return vad_path
    if pre_path.exists():
        return pre_path
    if vad_path.exists():
        return vad_path
    return None


def _write_sample_wav(input_wav: Path, output_wav: Path, sample_sec: float) -> None:
    audio, sr = sf.read(str(input_wav), always_2d=False)
    if audio is None or sr is None:
        raise SystemExit(f"Failed to read audio: {input_wav}")
    max_samples = int(max(sample_sec, 0.0) * sr)
    if max_samples > 0 and len(audio) > max_samples:
        audio = audio[:max_samples]
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav), audio, sr)


def _count_speakers_from_diarization(diar_path: Path) -> Optional[int]:
    if not diar_path.exists():
        return None
    data = json.loads(diar_path.read_text(encoding="utf-8"))
    segments = data.get("segments", []) if isinstance(data, dict) else []
    speakers = {seg.get("speaker") for seg in segments if isinstance(seg, dict)}
    speakers = {s for s in speakers if s}
    return len(speakers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full audio -> report pipeline.")
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input audio file path (default: latest file in audio_uploads when running preprocess).",
    )
    parser.add_argument("--origin", type=str, default=None, help="Override origin stem.")
    parser.add_argument("--title", type=str, default=None, help="Override report title.")
    parser.add_argument("--job-dir", type=str, default=None, help="Optional job output dir for status/log/zip.")
    parser.add_argument("--zip-name", type=str, default="result.zip")

    parser.add_argument("--preprocess-output", type=str, default=None)
    parser.add_argument("--stt-output", type=str, default=None)
    parser.add_argument("--summary-output", type=str, default=None)
    parser.add_argument("--report-output", type=str, default=None)
    parser.add_argument("--db-path", type=str, default=None)

    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-stt", action="store_true")
    parser.add_argument("--skip-diarization", action="store_true")
    parser.add_argument(
        "--run-diarization",
        action="store_false",
        dest="skip_diarization",
        help="Force run diarization (default: enabled).",
    )
    parser.add_argument("--skip-postprocess", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-embed", action="store_true")

    # preprocess options
    parser.add_argument("--max-segment-sec", type=float, default=None)
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--no-apply-vad", action="store_true")
    parser.add_argument("--vad-threshold", type=float, default=None)
    parser.add_argument("--vad-min-speech-sec", type=float, default=None)
    parser.add_argument("--vad-min-silence-sec", type=float, default=None)
    parser.add_argument("--vad-speech-pad-sec", type=float, default=None)

    # stt options
    parser.add_argument("--prefer-vad", dest="prefer_vad", action="store_true")
    parser.add_argument("--no-prefer-vad", dest="prefer_vad", action="store_false")
    parser.set_defaults(prefer_vad=None)
    parser.add_argument("--stt-model", type=str, default="medium")
    parser.add_argument("--stt-device", type=str, default=None)
    parser.add_argument("--stt-language", type=str, default=None)
    parser.add_argument("--stt-task", type=str, default=None)
    parser.add_argument("--stt-beam-size", type=int, default=None)
    parser.add_argument("--stt-temperature", type=float, default=None)
    parser.add_argument("--stt-best-of", type=int, default=None)
    parser.add_argument("--stt-initial-prompt", type=str, default=None)
    parser.add_argument("--stt-include-tokens", action="store_true")
    parser.add_argument("--stt-no-condition-on-previous-text", action="store_true")
    parser.add_argument("--stt-fp16", dest="stt_fp16", action="store_true")
    parser.add_argument("--stt-no-fp16", dest="stt_fp16", action="store_false")
    parser.set_defaults(stt_fp16=None)

    # diarization options
    parser.add_argument("--diar-device", type=str, default=None)
    parser.add_argument("--diar-num-speakers", type=int, default=None)
    parser.add_argument("--diar-min-speakers", type=int, default=None)
    parser.add_argument("--diar-max-speakers", type=int, default=None)
    parser.add_argument("--diar-min-overlap-ratio", type=float, default=None)
    parser.add_argument(
        "--diar-auto",
        action="store_true",
        help="Run diarization only if a short probe detects multiple speakers.",
    )
    parser.add_argument("--diar-sample-sec", type=float, default=180.0)
    parser.add_argument("--diar-auto-min-speakers", type=int, default=2)

    # postprocess options
    parser.add_argument("--post-keep-fillers", action="store_true")
    parser.add_argument("--post-min-len-keep-fillers", type=int, default=None)
    parser.add_argument("--post-drop-empty", action="store_true")
    parser.add_argument("--post-no-merge-speaker", action="store_true")
    parser.add_argument("--post-merge-gap", type=float, default=None)
    parser.add_argument("--post-max-chars-per-block", type=int, default=None)

    # summary options
    parser.add_argument("--summary-model", type=str, default=None)
    parser.add_argument("--summary-device-map", type=str, default=None)
    parser.add_argument("--summary-device", type=str, default=None)
    parser.add_argument("--detail-level", type=str, default=None)
    parser.add_argument("--summary-fast", action="store_true")
    parser.add_argument("--summary-profile", action="store_true")
    parser.add_argument("--max-chunk-tokens", type=int, default=None)
    parser.add_argument("--map-max-new-tokens", type=int, default=None)
    parser.add_argument("--reduce-max-new-tokens", type=int, default=None)
    parser.add_argument("--summary-temperature", type=float, default=None)
    parser.add_argument("--summary-top-p", type=float, default=None)

    # quality check options
    parser.add_argument("--qc-min-summary-chars", type=int, default=None)
    parser.add_argument("--qc-max-summary-chars", type=int, default=None)
    parser.add_argument("--qc-fill-empty-action-items", action="store_true")

    # export options
    parser.add_argument("--export-docx", action="store_true")
    parser.add_argument("--export-xlsx", action="store_true")

    # embed options
    parser.add_argument("--embed-collection", type=str, default=None)
    parser.add_argument("--embed-model", type=str, default=None)
    parser.add_argument("--embed-batch-size", type=int, default=None)
    parser.add_argument("--embed-device", type=str, default=None)
    parser.add_argument("--embed-reset", action="store_true")

    parser.set_defaults(skip_diarization=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    preprocess_root = Path(args.preprocess_output) if args.preprocess_output else DEFAULT_PREPROCESS_DIR
    stt_root = Path(args.stt_output) if args.stt_output else DEFAULT_STT_DIR
    summary_root = Path(args.summary_output) if args.summary_output else DEFAULT_SUMMARY_DIR
    report_root = Path(args.report_output) if args.report_output else DEFAULT_REPORT_DIR
    db_path = Path(args.db_path) if args.db_path else DEFAULT_DB_DIR

    # If no input path is provided and we will run preprocess,
    # default to the latest uploaded audio file so origin follows the source.
    if (not args.input) and (not args.skip_preprocess):
        latest_audio = _find_latest_audio(DEFAULT_UPLOAD_DIR)
        if latest_audio:
            args.input = str(latest_audio)

    origin = _resolve_origin(args.origin, args.input, preprocess_root, stt_root, summary_root)
    display_title = args.title or _make_display_title(args.input)

    prefer_vad = args.prefer_vad
    if prefer_vad is None:
        prefer_vad = True if args.diar_auto else False

    job_dir = Path(args.job_dir).resolve() if args.job_dir else None
    log_path = job_dir / "pipeline.log" if job_dir else None
    status_path = _init_status(job_dir, args.input) if job_dir else None

    if not args.skip_preprocess:
        cmd = [PYTHON_BIN, str(SCRIPT_PREPROCESS)]
        if args.input:
            cmd.append(args.input)
        _append_arg(cmd, "--origin", origin)
        _append_arg(cmd, "--output-dir", preprocess_root)
        _append_arg(cmd, "--max-segment-sec", args.max_segment_sec)
        _append_flag(cmd, "--no-vad", args.no_vad)
        _append_flag(cmd, "--no-apply-vad", args.no_apply_vad)
        _append_arg(cmd, "--vad-threshold", args.vad_threshold)
        _append_arg(cmd, "--vad-min-speech-sec", args.vad_min_speech_sec)
        _append_arg(cmd, "--vad-min-silence-sec", args.vad_min_silence_sec)
        _append_arg(cmd, "--vad-speech-pad-sec", args.vad_speech_pad_sec)
        if status_path:
            _update_status_stage(status_path, "preprocess")
        _run(cmd, log_path=log_path)
        if status_path:
            _append_status_stage(status_path, "preprocess", cmd, 0)

        if origin is None:
            origin = _resolve_origin(args.origin, args.input, preprocess_root, stt_root, summary_root)

    if not args.skip_stt:
        cmd = [PYTHON_BIN, str(SCRIPT_STT)]
        stt_input = None
        if args.input and args.skip_preprocess:
            stt_input = args.input
        elif origin:
            stt_input = str(preprocess_root / origin)
        if not stt_input:
            raise SystemExit("STT input not resolved. Provide input file or run preprocess.")
        cmd.append(stt_input)
        _append_arg(cmd, "--output-root", stt_root)
        if prefer_vad is True:
            cmd.append("--prefer-vad")
        elif prefer_vad is False:
            cmd.append("--no-prefer-vad")
        _append_arg(cmd, "--model", args.stt_model)
        _append_arg(cmd, "--device", args.stt_device)
        _append_arg(cmd, "--language", args.stt_language)
        _append_arg(cmd, "--task", args.stt_task)
        if args.stt_fp16 is True:
            cmd.append("--fp16")
        elif args.stt_fp16 is False:
            cmd.append("--no-fp16")
        _append_arg(cmd, "--beam-size", args.stt_beam_size)
        _append_arg(cmd, "--temperature", args.stt_temperature)
        _append_arg(cmd, "--best-of", args.stt_best_of)
        _append_arg(cmd, "--initial-prompt", args.stt_initial_prompt)
        _append_flag(cmd, "--include-tokens", args.stt_include_tokens)
        _append_flag(cmd, "--no-condition-on-previous-text", args.stt_no_condition_on_previous_text)
        if status_path:
            _update_status_stage(status_path, "stt")
        _run(cmd, log_path=log_path)
        if status_path:
            _append_status_stage(status_path, "stt", cmd, 0)

    if not args.skip_diarization:
        run_diarization = True
        if args.diar_auto:
            if not origin:
                origin = _resolve_origin(args.origin, args.input, preprocess_root, stt_root, summary_root)
            probe_input = None
            if origin:
                probe_input = _resolve_preprocessed_wav(preprocess_root, origin, prefer_vad)
            if probe_input and probe_input.exists():
                with tempfile.TemporaryDirectory(prefix="diar_probe_") as probe_dir:
                    probe_dir_path = Path(probe_dir)
                    suffix = "_preprocessed_vad.wav" if prefer_vad else "_preprocessed.wav"
                    sample_wav = probe_dir_path / f"{origin}{suffix}"
                    _write_sample_wav(probe_input, sample_wav, args.diar_sample_sec)
                    probe_out = probe_dir_path / "stt"
                    probe_cmd = [
                        PYTHON_BIN,
                        str(SCRIPT_DIARIZATION),
                        str(sample_wav),
                        "--output-root",
                        str(probe_out),
                    ]
                    if prefer_vad is True:
                        probe_cmd.append("--prefer-vad")
                    elif prefer_vad is False:
                        probe_cmd.append("--no-prefer-vad")
                    _append_arg(probe_cmd, "--device", args.diar_device)
                    _append_arg(probe_cmd, "--num-speakers", args.diar_num_speakers)
                    _append_arg(probe_cmd, "--min-speakers", args.diar_min_speakers)
                    _append_arg(probe_cmd, "--max-speakers", args.diar_max_speakers)
                    _append_arg(probe_cmd, "--min-overlap-ratio", args.diar_min_overlap_ratio)
                    _run(probe_cmd, log_path=log_path)
                    diar_path = probe_out / origin / f"{origin}_diarization.json"
                    speaker_count = _count_speakers_from_diarization(diar_path)
                    if speaker_count is not None and speaker_count < args.diar_auto_min_speakers:
                        run_diarization = False

        if run_diarization:
            cmd = [PYTHON_BIN, str(SCRIPT_DIARIZATION)]
            diar_input = None
            if args.input and args.skip_preprocess:
                diar_input = args.input
            elif origin:
                diar_input = str(preprocess_root / origin)
            if not diar_input:
                raise SystemExit("Diarization input not resolved. Provide input file or run preprocess.")
            cmd.append(diar_input)
            _append_arg(cmd, "--output-root", stt_root)
            if prefer_vad is True:
                cmd.append("--prefer-vad")
            elif prefer_vad is False:
                cmd.append("--no-prefer-vad")
            _append_arg(cmd, "--device", args.diar_device)
            _append_arg(cmd, "--num-speakers", args.diar_num_speakers)
            _append_arg(cmd, "--min-speakers", args.diar_min_speakers)
            _append_arg(cmd, "--max-speakers", args.diar_max_speakers)
            _append_arg(cmd, "--min-overlap-ratio", args.diar_min_overlap_ratio)
            if status_path:
                _update_status_stage(status_path, "diarization")
            _run(cmd, log_path=log_path)
            if status_path:
                _append_status_stage(status_path, "diarization", cmd, 0)
        else:
            if status_path:
                _append_status_stage(status_path, "diarization", ["auto-skip"], 0)

    if origin is None:
        origin = _resolve_origin(args.origin, args.input, preprocess_root, stt_root, summary_root)
    if not origin:
        raise SystemExit("Origin not resolved. Provide --origin or input path.")

    if not args.skip_postprocess:
        cmd = [PYTHON_BIN, str(SCRIPT_POSTPROCESS), origin]
        _append_arg(cmd, "--stt-root", stt_root)
        _append_flag(cmd, "--keep-fillers", args.post_keep_fillers)
        _append_arg(cmd, "--min-len-keep-fillers", args.post_min_len_keep_fillers)
        _append_flag(cmd, "--drop-empty", args.post_drop_empty)
        _append_flag(cmd, "--no-merge-speaker", args.post_no_merge_speaker)
        _append_arg(cmd, "--merge-gap", args.post_merge_gap)
        _append_arg(cmd, "--max-chars-per-block", args.post_max_chars_per_block)
        if status_path:
            _update_status_stage(status_path, "postprocess")
        _run(cmd, log_path=log_path)
        if status_path:
            _append_status_stage(status_path, "postprocess", cmd, 0)

    if not args.skip_summary:
        cmd = [PYTHON_BIN, str(SCRIPT_SUMMARY), origin]
        _append_arg(cmd, "--stt-root", stt_root)
        _append_arg(cmd, "--output-root", summary_root)
        _append_arg(cmd, "--model", args.summary_model)
        _append_arg(cmd, "--device-map", args.summary_device_map)
        _append_arg(cmd, "--device", args.summary_device)
        _append_arg(cmd, "--detail-level", args.detail_level)
        _append_flag(cmd, "--fast", args.summary_fast)
        _append_flag(cmd, "--profile", args.summary_profile)
        _append_arg(cmd, "--max-chunk-tokens", args.max_chunk_tokens)
        _append_arg(cmd, "--map-max-new-tokens", args.map_max_new_tokens)
        _append_arg(cmd, "--reduce-max-new-tokens", args.reduce_max_new_tokens)
        _append_arg(cmd, "--temperature", args.summary_temperature)
        _append_arg(cmd, "--top-p", args.summary_top_p)
        if status_path:
            _update_status_stage(status_path, "summary")
        _run(cmd, log_path=log_path)
        if status_path:
            _append_status_stage(status_path, "summary", cmd, 0)

    if not args.skip_quality:
        cmd = [PYTHON_BIN, str(SCRIPT_QC), origin]
        _append_arg(cmd, "--summary-root", summary_root)
        _append_arg(cmd, "--min-summary-chars", args.qc_min_summary_chars)
        _append_arg(cmd, "--max-summary-chars", args.qc_max_summary_chars)
        _append_flag(cmd, "--fill-empty-action-items", args.qc_fill_empty_action_items)
        if status_path:
            _update_status_stage(status_path, "quality_check")
        _run(cmd, log_path=log_path)
        if status_path:
            _append_status_stage(status_path, "quality_check", cmd, 0)

    if not args.skip_export:
        cmd = [PYTHON_BIN, str(SCRIPT_EXPORT), origin]
        _append_arg(cmd, "--summary-root", summary_root)
        _append_arg(cmd, "--report-root", report_root)
        _append_arg(cmd, "--title", display_title)
        _append_flag(cmd, "--docx", args.export_docx)
        _append_flag(cmd, "--xlsx", args.export_xlsx)
        if status_path:
            _update_status_stage(status_path, "export_report")
        _run(cmd, log_path=log_path)
        if status_path:
            _append_status_stage(status_path, "export_report", cmd, 0)

    if not args.skip_embed:
        cmd = [PYTHON_BIN, str(SCRIPT_EMBED), origin]
        _append_arg(cmd, "--summary-root", summary_root)
        _append_arg(cmd, "--db-path", db_path)
        _append_arg(cmd, "--collection", args.embed_collection)
        _append_arg(cmd, "--model", args.embed_model)
        _append_arg(cmd, "--batch-size", args.embed_batch_size)
        _append_arg(cmd, "--device", args.embed_device)
        _append_flag(cmd, "--reset", args.embed_reset)
        if status_path:
            _update_status_stage(status_path, "report_embed")
        _run(cmd, log_path=log_path)
        if status_path:
            _append_status_stage(status_path, "report_embed", cmd, 0)

    if status_path and job_dir:
        zip_path = _make_result_zip(job_dir, args.zip_name)
        _finish_status(status_path, result_zip=str(zip_path.name), error=None)

    print("[stt_run] done")


if __name__ == "__main__":
    main()