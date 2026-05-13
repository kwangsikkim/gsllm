#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_SUMMARY_DIR = Path(__file__).resolve().parent / "summary_output"


@dataclass
class Verdict:
    PASS: str = "PASS"
    WARN: str = "WARN"
    FAIL: str = "FAIL"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_latest_dir(root: Path) -> Path:
    if not root.exists():
        raise SystemExit(f"Summary output root not found: {root}")
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise SystemExit(f"No summary folders found in {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def pick_latest_file(files: List[Path]) -> Optional[Path]:
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def load_summary_json(root: Path, origin: Optional[str]) -> Path:
    if origin:
        target = root / origin / f"{origin}_summary.json"
        if target.exists():
            return target
        raise SystemExit(f"No summary json found in {root / origin}")

    latest_dir = find_latest_dir(root)
    candidates = list(latest_dir.glob("*_summary.json"))
    picked = pick_latest_file(candidates)
    if picked:
        return picked
    raise SystemExit(f"No summary json found in {latest_dir}")


def origin_from_summary_filename(name: str) -> str:
    if name.endswith("_summary.json"):
        return name[: -len("_summary.json")]
    return Path(name).stem


def ensure_action_item_schema(item: Any) -> Dict[str, str]:
    item = item if isinstance(item, dict) else {}
    return {
        "who": (item.get("who") or "미정"),
        "what": (item.get("what") or "미정"),
        "due": (item.get("due") or "미정"),
    }


def check_action_items(action_items: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    issues: List[Dict[str, Any]] = []
    fixed_items: List[Dict[str, str]] = []

    items = action_items if isinstance(action_items, list) else []
    for idx, item in enumerate(items):
        fixed = ensure_action_item_schema(item)
        fixed_items.append(fixed)
        missing = [k for k, v in fixed.items() if v == "미정"]
        if missing:
            issues.append({"index": idx, "missing": missing, "message": "action_item 필드 누락"})

    return issues, fixed_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality check for summary outputs.")
    parser.add_argument("origin", nargs="?", default=None)
    parser.add_argument("--summary-root", type=str, default=None)
    parser.add_argument("--min-summary-chars", type=int, default=400)
    parser.add_argument("--max-summary-chars", type=int, default=6000)
    parser.add_argument("--fill-empty-action-items", action="store_true")
    return parser.parse_args()


def write_qc_report(report_path: Path, qc_report: Dict[str, Any]) -> None:
    report_path.write_text(json.dumps(qc_report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary_root = Path(args.summary_root) if args.summary_root else DEFAULT_SUMMARY_DIR
    summary_path = load_summary_json(summary_root, args.origin)

    output_dir = summary_path.parent
    origin_stem = origin_from_summary_filename(summary_path.name)
    report_path = output_dir / f"{origin_stem}_qc_report.json"

    try:
        raw = summary_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as exc:
        qc_report = {
            "created_at": utc_now_iso(),
            "source_summary": str(summary_path),
            "verdict": Verdict.FAIL,
            "warnings": ["요약 JSON을 읽거나 파싱하는 데 실패했습니다."],
            "errors": [{"message": str(exc)}],
            "retry_suggested": True,
        }
        write_qc_report(report_path, qc_report)
        print(f"Saved: {report_path}")
        return

    summary_obj = data.get("summary", {}) if isinstance(data, dict) else {}
    summary_text = ""
    action_items: Any = []

    if isinstance(summary_obj, dict):
        summary_text = (summary_obj.get("summary") or "").strip()
        action_items = summary_obj.get("action_items") or []
    else:
        summary_text = ""
        action_items = []

    warnings: List[str] = []
    errors: List[Dict[str, Any]] = []
    retry_suggested = False

    summary_chars = len(summary_text)

    if summary_chars == 0:
        warnings.append("요약 텍스트가 비어 있습니다.")
        retry_suggested = True
    if summary_chars < args.min_summary_chars:
        warnings.append("요약이 너무 짧습니다.")
        retry_suggested = True
    if summary_chars > args.max_summary_chars:
        warnings.append("요약이 너무 깁니다.")
        retry_suggested = True

    issues, fixed_items = check_action_items(action_items)

    if not isinstance(action_items, list):
        warnings.append("action_items가 리스트 형식이 아닙니다.")
        retry_suggested = True

    if (not action_items) or (isinstance(action_items, list) and len(action_items) == 0):
        warnings.append("액션 아이템이 없습니다. '없음' 표시 필요.")
        if args.fill_empty_action_items:
            fixed_items = [{"who": "없음", "what": "없음", "due": "없음"}]

    if issues:
        warnings.append("액션 아이템 일부에 필드 누락이 있습니다.")
        retry_suggested = True

    verdict = Verdict.PASS
    if warnings:
        verdict = Verdict.WARN

    qc_report = {
        "created_at": utc_now_iso(),
        "source_summary": str(summary_path),
        "verdict": verdict,
        "metrics": {
            "summary_chars": summary_chars,
            "action_items_count": len(action_items) if isinstance(action_items, list) else 0,
            "action_item_issue_count": len(issues),
        },
        "warnings": warnings,
        "errors": errors,
        "retry_suggested": retry_suggested,
        "action_item_issues": issues,
        "action_items_fixed": fixed_items,
    }

    write_qc_report(report_path, qc_report)
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
