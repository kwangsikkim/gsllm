#!/usr/bin/env python3
import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_STT_DIR = Path(__file__).resolve().parent / "stt_output"
DEFAULT_SUMMARY_DIR = Path(__file__).resolve().parent / "summary_output"


# ----------------------------
# Config
# ----------------------------
@dataclass
class SummaryConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "cuda"  # used when device_map is not "auto"
    device_map: str = "auto"  # "auto" (recommended) or "none"
    max_chunk_tokens: int = 3600  # leave room for prompt/template overhead
    map_max_new_tokens: int = 1400
    reduce_max_new_tokens: int = 2200
    temperature: float = 0.2
    top_p: float = 0.9
    detail_level: str = "detailed"  # brief|normal|detailed

    # reduce compression
    max_points: int = 18
    max_actions: int = 12
    max_decisions: int = 12
    max_risks: int = 10
    max_issues: int = 10
    max_open_questions: int = 12
    max_next_steps: int = 14
    max_agenda: int = 14
    max_discussion_flow: int = 20
    max_details: int = 24

    reduce_input_budget_chars: int = 140_000
    map_retry: int = 1  # retry once if json parsing fails


# ----------------------------
# IO helpers
# ----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_latest_dir(root: Path) -> Path:
    if not root.exists():
        raise SystemExit(f"STT output root not found: {root}")
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise SystemExit(f"No STT folders found in {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_clean_transcript(stt_dir: Path, origin: Optional[str]) -> Path:
    """
    - If origin specified: stt_dir/origin/origin_clean_transcript.json
    - Else: pick latest folder under stt_dir, then pick newest *_clean_transcript.json
    """
    if origin:
        target = stt_dir / origin / f"{origin}_clean_transcript.json"
        if target.exists():
            return target
        raise SystemExit(f"No clean transcript found in {stt_dir / origin}")

    latest = find_latest_dir(stt_dir)
    candidates = list(latest.glob("*_clean_transcript.json"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    raise SystemExit(f"No clean transcript found in {latest}")


# ----------------------------
# Transcript -> text
# ----------------------------
def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def build_text_from_segments(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = seg.get("speaker")
        start = _safe_float(seg.get("start"))
        end = _safe_float(seg.get("end"))
        if speaker:
            if start is not None and end is not None:
                lines.append(f"{speaker} [{start:.2f}-{end:.2f}]: {text}")
            else:
                lines.append(f"{speaker}: {text}")
        else:
            lines.append(text)
    return "\n".join(lines).strip()


# ----------------------------
# Chunking
# ----------------------------
def chunk_by_tokens(text: str, tokenizer, max_tokens: int) -> List[str]:
    """
    Token-based chunking using paragraph (line) boundaries.
    """
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for para in paragraphs:
        tokens = len(tokenizer.encode(para, add_special_tokens=False))
        if current and current_tokens + tokens > max_tokens:
            chunks.append("\n".join(current))
            current = [para]
            current_tokens = tokens
        else:
            current.append(para)
            current_tokens += tokens

    if current:
        chunks.append("\n".join(current))
    return chunks


# ----------------------------
# JSON extraction (robust)
# ----------------------------
def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust JSON extraction:
    1) Try code fence ```json ... ```
    2) Try non-greedy object matches, test each with json.loads
    """
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    candidates: List[str] = []
    if fence:
        candidates.append(fence.group(1).strip())

    candidates += re.findall(r"\{[\s\S]*?\}", text)

    best = None
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                # Prefer objects that look like our schema
                if "summary" in obj and ("key_points" in obj or "agenda" in obj):
                    return obj
                if best is None:
                    best = obj
        except Exception:
            continue
    return best


def _ensure_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _ensure_str_list(x: Any) -> List[str]:
    xs = _ensure_list(x)
    out: List[str] = []
    for item in xs:
        if item is None:
            continue
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
        else:
            s = str(item).strip()
            if s:
                out.append(s)
    return out


def _ensure_action_items(x: Any) -> List[Dict[str, str]]:
    xs = _ensure_list(x)
    out: List[Dict[str, str]] = []
    for item in xs:
        if isinstance(item, dict):
            out.append(
                {
                    "who": str(item.get("who") or "미정"),
                    "what": str(item.get("what") or "미정"),
                    "due": str(item.get("due") or "미정"),
                }
            )
        else:
            # allow string fallback
            s = str(item).strip()
            if s:
                out.append({"who": "미정", "what": s, "due": "미정"})
    return out


def _ensure_speaker_summary(x: Any) -> List[Dict[str, str]]:
    xs = _ensure_list(x)
    out: List[Dict[str, str]] = []
    for item in xs:
        if isinstance(item, dict):
            out.append(
                {
                    "speaker": str(item.get("speaker") or "UNKNOWN"),
                    "summary": str(item.get("summary") or "-").strip(),
                }
            )
        else:
            s = str(item).strip()
            if s:
                out.append({"speaker": "UNKNOWN", "summary": s})
    return out


def normalize_summary_schema(obj: Optional[Dict[str, Any]], fallback_text: str) -> Dict[str, Any]:
    """
    New schema includes:
      - agenda: string[]
      - discussion_flow: string[]
      - details: string[]
    """
    if not isinstance(obj, dict):
        obj = {}

    summary_text = (obj.get("summary") or "").strip()

    # If summary itself is a JSON blob, parse and merge
    if summary_text.lstrip().startswith("{") and '"summary"' in summary_text:
        try:
            parsed_obj = json.loads(summary_text)
        except Exception:
            parsed_obj = None
        if isinstance(parsed_obj, dict):
            obj = {**obj, **parsed_obj}
            summary_text = (obj.get("summary") or "").strip()

    def _extract_summary_from_text(text: str) -> str:
        if not text:
            return ""
        candidate = parse_json_from_text(text)
        if isinstance(candidate, dict):
            extracted = (candidate.get("summary") or "").strip()
            if extracted:
                return extracted
        return text.strip()

    if not summary_text:
        summary_text = _extract_summary_from_text(fallback_text or "")
    elif summary_text.lstrip().startswith("{") or '"summary"' in summary_text:
        summary_text = _extract_summary_from_text(summary_text)

    normalized = {
        "summary": summary_text,
        "context": (obj.get("context") or "").strip(),

        # NEW
        "agenda": _ensure_str_list(obj.get("agenda")),
        "discussion_flow": _ensure_str_list(obj.get("discussion_flow")),
        "details": _ensure_str_list(obj.get("details")),

        "key_points": _ensure_str_list(obj.get("key_points")),
        "decisions": _ensure_str_list(obj.get("decisions")),
        "action_items": _ensure_action_items(obj.get("action_items")),
        "risks": _ensure_str_list(obj.get("risks")),
        "issues": _ensure_str_list(obj.get("issues")),
        "open_questions": _ensure_str_list(obj.get("open_questions")),
        "next_steps": _ensure_str_list(obj.get("next_steps")),
        "speaker_summary": _ensure_speaker_summary(obj.get("speaker_summary")),
    }

    return normalized


def apply_speaker_numbering(summary: Dict[str, Any], enforce: bool) -> Dict[str, Any]:
    if not enforce:
        return summary
    items = summary.get("speaker_summary", [])
    if not isinstance(items, list) or not items:
        return summary

    mapping: Dict[str, str] = {}
    next_idx = 1
    numbered: List[Dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        speaker_raw = str(item.get("speaker") or "").strip()
        if speaker_raw not in mapping:
            mapping[speaker_raw] = f"화자{next_idx}"
            next_idx += 1
        numbered.append(
            {
                "speaker": mapping[speaker_raw],
                "summary": str(item.get("summary") or "-").strip(),
            }
        )

    summary["speaker_summary"] = numbered
    return summary


# ----------------------------
# Prompting (detail control)
# ----------------------------
def _detail_specs(detail_level: str) -> Dict[str, str]:
    detail_level = (detail_level or "normal").lower()
    if detail_level == "brief":
        return {
            "summary_len": "6~10줄",
            "agenda": "최대 6개",
            "flow": "최대 8개",
            "details": "최대 10개",
            "key_points": "최대 8개",
            "speaker": "화자별 요약은 1문장씩",
            "extras": "불필요한 배경 설명은 줄이고 결론/핵심 설명 위주",
        }
    if detail_level == "detailed":
        return {
            "summary_len": "2~3문단 (또는 12~18문장)",
            "agenda": "최대 10~14개",
            "flow": "최대 12~20개",
            "details": "최대 14~24개",
            "key_points": "최대 12~18개",
            "speaker": "화자별 요약은 2~3문장",
            "extras": "가능하면 숫자/단위/고유명사/원리/예시/비유를 포함",
        }
    return {
        "summary_len": "1~2문단",
        "agenda": "최대 8~12개",
        "flow": "최대 10~16개",
        "details": "최대 12~18개",
        "key_points": "최대 10~14개",
        "speaker": "화자별 요약은 1~2문장",
        "extras": "핵심 설명과 흐름을 균형 있게",
    }


def make_map_prompt(chunk: str, detail_level: str) -> str:
    spec = _detail_specs(detail_level)

    # 핵심: '키워드 나열 금지', '무엇/왜/어떻게' 강제, '구체 설명' 강제
    return (
        "너는 한국어 회의/강의/대화 요약 전문가다.\n"
        "반드시 'JSON 객체만' 출력해라. (설명/문장/마크다운/코드블록 금지)\n"
        "아래 스키마를 정확히 만족해야 한다.\n\n"
        "스키마:\n"
        "{\n"
        '  "summary": string,\n'
        '  "context": string,\n'
        '  "agenda": string[],\n'
        '  "discussion_flow": string[],\n'
        '  "details": string[],\n'
        '  "key_points": string[],\n'
        '  "decisions": string[],\n'
        '  "action_items": [{"who": string, "what": string, "due": string}],\n'
        '  "risks": string[],\n'
        '  "issues": string[],\n'
        '  "open_questions": string[],\n'
        '  "next_steps": string[],\n'
        '  "speaker_summary": [{"speaker": string, "summary": string}]\n'
        "}\n\n"
        f"작성 규칙(디테일 레벨={detail_level}):\n"
        f"- summary: {spec['summary_len']}\n"
        f"- agenda: {spec['agenda']}\n"
        f"- discussion_flow: {spec['flow']}\n"
        f"- details: {spec['details']}\n"
        f"- key_points: {spec['key_points']}\n"
        f"- speaker_summary: {spec['speaker']}\n"
        f"- {spec['extras']}\n\n"
        "매우 중요:\n"
        "- '키워드/주제 이름만 나열' 금지. 각 항목은 반드시 **무엇/왜/어떻게** 중 최소 1개를 포함해 구체적으로.\n"
        "- discussion_flow는 대화/설명의 전개 순서대로 'A를 설명→B로 연결→C 예시'처럼 흐름이 보이게.\n"
        "- details는 실제로 설명된 개념/원리/비유/수치/예시를 구체적으로 적어라.\n"
        "- action_items의 who/what/due는 가능한 구체적으로. 없으면 '미정'.\n"
        "- context는 목적/배경을 1~2줄.\n\n"
        f"전사:\n{chunk}\n"
    )


def make_reduce_prompt(items: List[Dict[str, Any]], detail_level: str) -> str:
    spec = _detail_specs(detail_level)
    return (
        "너는 한국어 회의/강의/대화 요약 전문가다.\n"
        "아래 입력(청크 요약 JSON 배열)을 통합해 최종 'JSON 객체만' 출력해라.\n"
        "설명/마크다운/코드블록 금지. 오직 JSON.\n\n"
        "최종 스키마:\n"
        "{\n"
        '  "summary": string,\n'
        '  "context": string,\n'
        '  "agenda": string[],\n'
        '  "discussion_flow": string[],\n'
        '  "details": string[],\n'
        '  "key_points": string[],\n'
        '  "decisions": string[],\n'
        '  "action_items": [{"who": string, "what": string, "due": string}],\n'
        '  "risks": string[],\n'
        '  "issues": string[],\n'
        '  "open_questions": string[],\n'
        '  "next_steps": string[],\n'
        '  "speaker_summary": [{"speaker": string, "summary": string}]\n'
        "}\n\n"
        f"작성 규칙(디테일 레벨={detail_level}):\n"
        f"- summary: {spec['summary_len']}\n"
        f"- agenda: {spec['agenda']}\n"
        f"- discussion_flow: {spec['flow']}\n"
        f"- details: {spec['details']}\n"
        f"- key_points: {spec['key_points']}\n"
        f"- speaker_summary: {spec['speaker']}\n"
        "- 중복 제거. 서로 충돌하면 더 그럴듯한 쪽으로 정리.\n"
        "- topic 나열 금지: 반드시 설명/정의/원리/예시를 포함.\n"
        "- action_items는 동일 항목 병합, 기한/담당 없으면 '미정'.\n\n"
        f"입력:\n{json.dumps(items, ensure_ascii=False)}\n"
    )


# ----------------------------
# Reduce input compression
# ----------------------------
def _trim_list(xs: Any, max_n: int) -> List[Any]:
    if not isinstance(xs, list):
        return []
    return xs[:max_n]


def compress_map_results(map_results: List[Dict[str, Any]], cfg: SummaryConfig) -> List[Dict[str, Any]]:
    """
    Reduce prompt can explode when many chunks exist.
    Keep essentials and cap list sizes per chunk.
    """
    compressed: List[Dict[str, Any]] = []
    total_chars = 0

    for r in map_results:
        item = {
            "summary": (r.get("summary") or "")[:2200],
            "context": (r.get("context") or "")[:700],

            # NEW
            "agenda": _trim_list(r.get("agenda"), cfg.max_agenda),
            "discussion_flow": _trim_list(r.get("discussion_flow"), cfg.max_discussion_flow),
            "details": _trim_list(r.get("details"), cfg.max_details),

            "key_points": _trim_list(r.get("key_points"), cfg.max_points),
            "decisions": _trim_list(r.get("decisions"), cfg.max_decisions),
            "action_items": _trim_list(r.get("action_items"), cfg.max_actions),
            "risks": _trim_list(r.get("risks"), cfg.max_risks),
            "issues": _trim_list(r.get("issues"), cfg.max_issues),
            "open_questions": _trim_list(r.get("open_questions"), cfg.max_open_questions),
            "next_steps": _trim_list(r.get("next_steps"), cfg.max_next_steps),
            "speaker_summary": _trim_list(r.get("speaker_summary"), 12),
        }
        s = json.dumps(item, ensure_ascii=False)
        if total_chars + len(s) > cfg.reduce_input_budget_chars:
            break
        compressed.append(item)
        total_chars += len(s)

    return compressed


# ----------------------------
# Markdown rendering
# ----------------------------
def render_markdown(summary: Dict[str, Any]) -> str:
    def bullet(title: str, items: List[Any]) -> str:
        if not items:
            return f"## {title}\n- 없음\n"
        lines = [f"## {title}"]
        for item in items:
            if isinstance(item, dict):
                lines.append(f"- {json.dumps(item, ensure_ascii=False)}")
            else:
                lines.append(f"- {str(item)}")
        return "\n".join(lines) + "\n"

    md_parts: List[str] = []
    md_parts.append("# 요약\n")
    md_parts.append((summary.get("summary") or "").strip() + "\n")

    context = (summary.get("context") or "").strip()
    if context:
        md_parts.append("## 배경/목적\n")
        md_parts.append(context + "\n")

    # NEW sections
    md_parts.append(bullet("아젠다", summary.get("agenda", [])))
    md_parts.append(bullet("논의 흐름", summary.get("discussion_flow", [])))
    md_parts.append(bullet("구체 내용/설명", summary.get("details", [])))

    md_parts.append(bullet("핵심 포인트", summary.get("key_points", [])))
    md_parts.append(bullet("결정사항", summary.get("decisions", [])))

    action_items = summary.get("action_items", [])
    if action_items:
        lines = ["## 액션 아이템"]
        for item in action_items:
            if isinstance(item, dict):
                who = item.get("who") or "미정"
                what = item.get("what") or "-"
                due = item.get("due") or "미정"
                lines.append(f"- {who}: {what} (기한: {due})")
            else:
                lines.append(f"- {item}")
        md_parts.append("\n".join(lines) + "\n")
    else:
        md_parts.append("## 액션 아이템\n- 없음\n")

    md_parts.append(bullet("리스크", summary.get("risks", [])))
    md_parts.append(bullet("이슈", summary.get("issues", [])))
    md_parts.append(bullet("미해결 질문", summary.get("open_questions", [])))
    md_parts.append(bullet("다음 단계", summary.get("next_steps", [])))

    speaker_summary = summary.get("speaker_summary", [])
    if speaker_summary:
        lines = ["## 화자별 요약"]
        for item in speaker_summary:
            if isinstance(item, dict):
                speaker = item.get("speaker") or "UNKNOWN"
                ssum = item.get("summary") or "-"
                lines.append(f"- {speaker}: {ssum}")
        md_parts.append("\n".join(lines) + "\n")

    return "\n".join(md_parts).strip() + "\n"


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize transcript with Qwen2.5 14B Instruct (map-reduce)."
    )
    parser.add_argument("origin", nargs="?", default=None)
    parser.add_argument("--stt-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device-map", type=str, default="auto", choices=["auto", "none"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--detail-level", type=str, default="detailed", choices=["brief", "normal", "detailed"])

    parser.add_argument("--max-chunk-tokens", type=int, default=3600)
    parser.add_argument("--map-max-new-tokens", type=int, default=1400)
    parser.add_argument("--reduce-max-new-tokens", type=int, default=2200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)

    parser.add_argument("--max-points", type=int, default=18)
    parser.add_argument("--max-actions", type=int, default=12)
    parser.add_argument("--max-decisions", type=int, default=12)
    parser.add_argument("--max-risks", type=int, default=10)
    parser.add_argument("--max-issues", type=int, default=10)
    parser.add_argument("--max-open-questions", type=int, default=12)
    parser.add_argument("--max-next-steps", type=int, default=14)

    # NEW caps
    parser.add_argument("--max-agenda", type=int, default=14)
    parser.add_argument("--max-discussion-flow", type=int, default=20)
    parser.add_argument("--max-details", type=int, default=24)

    parser.add_argument("--reduce-input-budget-chars", type=int, default=140_000)
    parser.add_argument("--map-retry", type=int, default=1)
    return parser.parse_args()


# ----------------------------
# Model helpers
# ----------------------------
def _ensure_pad_token(tokenizer) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token


def _encode_messages(tokenizer, messages: List[Dict[str, str]], fallback_prompt: str):
    """
    Safely handle apply_chat_template returning either Tensor or BatchEncoding.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        enc = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        if isinstance(enc, dict):
            return enc["input_ids"]
        return enc
    return tokenizer(fallback_prompt, return_tensors="pt").input_ids


def _default_summary_skeleton(text: str) -> Dict[str, Any]:
    return {
        "summary": text.strip(),
        "context": "",
        "agenda": [],
        "discussion_flow": [],
        "details": [],
        "key_points": [],
        "decisions": [],
        "action_items": [],
        "risks": [],
        "issues": [],
        "open_questions": [],
        "next_steps": [],
        "speaker_summary": [],
    }


def _origin_stem_from_clean_filename(name: str) -> str:
    if name.endswith("_clean_transcript.json"):
        return name[: -len("_clean_transcript.json")]
    return Path(name).stem


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    args = parse_args()

    cfg = SummaryConfig(
        model_name=args.model,
        device=args.device,
        device_map=args.device_map,
        max_chunk_tokens=args.max_chunk_tokens,
        map_max_new_tokens=args.map_max_new_tokens,
        reduce_max_new_tokens=args.reduce_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        detail_level=args.detail_level,

        max_points=args.max_points,
        max_actions=args.max_actions,
        max_decisions=args.max_decisions,
        max_risks=args.max_risks,
        max_issues=args.max_issues,
        max_open_questions=args.max_open_questions,
        max_next_steps=args.max_next_steps,

        max_agenda=args.max_agenda,
        max_discussion_flow=args.max_discussion_flow,
        max_details=args.max_details,

        reduce_input_budget_chars=args.reduce_input_budget_chars,
        map_retry=args.map_retry,
    )

    stt_root = Path(args.stt_root) if args.stt_root else DEFAULT_STT_DIR
    transcript_path = load_clean_transcript(stt_root, args.origin)

    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    text = build_text_from_segments(segments)
    if not text:
        raise SystemExit("Empty transcript text.")
    has_speakers = any(
        isinstance(seg, dict) and (seg.get("speaker") or "").strip()
        for seg in segments
    )

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: transformers/torch. Install via `pip install transformers torch`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    _ensure_pad_token(tokenizer)

    if cfg.device_map == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype="auto",
        ).to(cfg.device)

    model.eval()

    chunks = chunk_by_tokens(text, tokenizer, cfg.max_chunk_tokens)
    if not chunks:
        raise SystemExit("No chunks generated from transcript.")

    map_results: List[Dict[str, Any]] = []

    for chunk in chunks:
        prompt = make_map_prompt(chunk, cfg.detail_level)
        messages = [
            {"role": "system", "content": "당신은 한국어 회의/강의 요약 전문가입니다."},
            {"role": "user", "content": prompt},
        ]

        input_ids = _encode_messages(tokenizer, messages, prompt).to(model.device)

        parsed: Optional[Dict[str, Any]] = None
        last_decoded = ""

        for attempt in range(cfg.map_retry + 1):
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=cfg.map_max_new_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    do_sample=cfg.temperature > 0,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            last_decoded = tokenizer.decode(
                output_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            parsed = parse_json_from_text(last_decoded)
            if parsed is not None:
                break

            if attempt < cfg.map_retry:
                retry_prompt = (
                    "방금 출력이 JSON이 아니었습니다. 반드시 JSON 객체만 출력하세요. "
                    "설명/마크다운/코드블록 없이 오직 JSON.\n\n" + prompt
                )
                messages = [
                    {"role": "system", "content": "당신은 한국어 회의/강의 요약 전문가입니다."},
                    {"role": "user", "content": retry_prompt},
                ]
                input_ids = _encode_messages(tokenizer, messages, retry_prompt).to(model.device)

        if parsed is None:
            parsed = _default_summary_skeleton(last_decoded)

        parsed = normalize_summary_schema(parsed, last_decoded)
        parsed = apply_speaker_numbering(parsed, enforce=not has_speakers)
        map_results.append(parsed)

    reduce_items = compress_map_results(map_results, cfg)
    reduce_prompt = make_reduce_prompt(reduce_items, cfg.detail_level)
    reduce_messages = [
        {"role": "system", "content": "당신은 한국어 회의/강의 요약 전문가입니다."},
        {"role": "user", "content": reduce_prompt},
    ]

    reduce_input_ids = _encode_messages(tokenizer, reduce_messages, reduce_prompt).to(model.device)

    with torch.no_grad():
        reduce_output_ids = model.generate(
            reduce_input_ids,
            max_new_tokens=cfg.reduce_max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=cfg.temperature > 0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    reduce_decoded = tokenizer.decode(
        reduce_output_ids[0][reduce_input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    summary_json = parse_json_from_text(reduce_decoded)
    if summary_json is None:
        summary_json = _default_summary_skeleton(reduce_decoded)
    summary_json = normalize_summary_schema(summary_json, reduce_decoded)
    summary_json = apply_speaker_numbering(summary_json, enforce=not has_speakers)

    created_at = utc_now_iso()
    origin_stem = _origin_stem_from_clean_filename(transcript_path.name)

    output_root = Path(args.output_root) if args.output_root else DEFAULT_SUMMARY_DIR
    output_dir = output_root / origin_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "created_at": created_at,
        "model_name": cfg.model_name,
        "detail_level": cfg.detail_level,
        "source_transcript": str(transcript_path),
        "summary": summary_json,
        "chunks": map_results,
    }

    summary_json_path = output_dir / f"{origin_stem}_summary.json"
    summary_md_path = output_dir / f"{origin_stem}_summary.md"

    summary_json_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md_path.write_text(render_markdown(summary_json), encoding="utf-8")

    print(f"Saved: {summary_json_path}")
    print(f"Saved: {summary_md_path}")


if __name__ == "__main__":
    main()
