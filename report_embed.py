#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_SUMMARY_DIR = Path(__file__).resolve().parent / "summary_output"
DEFAULT_DB_DIR = Path(__file__).resolve().parent / "chroma_db"
DEFAULT_COLLECTION = "reports"

SECTION_ARTIFACT_RE = re.compile(
    r"^\s*(#{1,6}\s+)?(요약|배경/목적|핵심 포인트|결정사항|액션 아이템|리스크|이슈|미해결 질문|다음 단계|화자별 요약|품질 점검)\s*$",
    re.IGNORECASE,
)
SUMMARY_VALUE_RE = re.compile(r'"summary"\s*:\s*"([^"]+)"', re.DOTALL)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def origin_from_summary_filename(name: str) -> str:
    if name.endswith("_summary.json"):
        return name[: -len("_summary.json")]
    return Path(name).stem


def find_latest_dir(root: Path) -> Path:
    if not root.exists():
        raise SystemExit(f"Summary output root not found: {root}")
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise SystemExit(f"No summary folders found in {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_summary_json(root: Path, origin: Optional[str]) -> Path:
    if origin:
        target = root / origin / f"{origin}_summary.json"
        if target.exists():
            return target
        raise SystemExit(f"No summary json found in {root / origin}")

    latest = find_latest_dir(root)
    candidates = list(latest.glob("*_summary.json"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    raise SystemExit(f"No summary json found in {latest}")


def load_qc_json(summary_path: Path) -> Optional[Path]:
    origin_stem = origin_from_summary_filename(summary_path.name)
    qc_path = summary_path.parent / f"{origin_stem}_qc_report.json"
    return qc_path if qc_path.exists() else None


def _parse_json_blob(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "summary" in value:
            return str(value.get("summary") or "")
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    t = text.strip()
    if not t:
        return ""

    if t.startswith("{") and '"summary"' in t:
        parsed = _parse_json_blob(t)
        if isinstance(parsed, dict) and "summary" in parsed:
            t = _as_text(parsed.get("summary")).strip()
        else:
            m = SUMMARY_VALUE_RE.search(t)
            if m:
                t = m.group(1).strip()

    lines: List[str] = []
    for raw in t.splitlines():
        line = raw.strip()
        if not line:
            continue
        if SECTION_ARTIFACT_RE.match(line):
            continue
        lines.append(line)
    return " ".join(lines).strip()


def _flatten_items(items: Any) -> List[str]:
    if items is None:
        return []
    if not isinstance(items, list):
        return []
    flattened: List[str] = []
    for item in items:
        if isinstance(item, dict):
            text = ", ".join(f"{k}: {v}" for k, v in item.items() if v is not None)
            if text:
                flattened.append(text)
            else:
                flattened.append(json.dumps(item, ensure_ascii=False))
            continue
        if isinstance(item, str):
            cleaned = _clean_text(item)
            if cleaned:
                flattened.append(cleaned)
            continue
        flattened.append(_clean_text(str(item)))
    return [x for x in flattened if x]


def _prune_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in meta.items() if v is not None}


def _collect_summary_docs(summary_data: Dict[str, Any], origin: str) -> List[Tuple[str, Dict[str, Any]]]:
    docs: List[Tuple[str, Dict[str, Any]]] = []
    summary_obj = summary_data.get("summary", {})
    if isinstance(summary_obj, str):
        parsed = _parse_json_blob(summary_obj)
        if isinstance(parsed, dict):
            summary_obj = parsed
        else:
            summary_obj = {"summary": summary_obj}
    if not isinstance(summary_obj, dict):
        summary_obj = {}

    def add_doc(text: str, section: str, field: str, idx: Optional[int] = None, chunk_idx: Optional[int] = None) -> None:
        cleaned = _clean_text(text)
        if not cleaned:
            return
        meta = _prune_metadata({
            "origin": origin,
            "section": section,
            "field": field,
            "idx": idx,
            "chunk_idx": chunk_idx,
            "created_at": summary_data.get("created_at") or utc_now_iso(),
        })
        docs.append((cleaned, meta))

    add_doc(_as_text(summary_obj.get("summary")), "summary", "summary")
    add_doc(_as_text(summary_obj.get("context")), "summary", "context")

    list_fields = [
        "agenda",
        "discussion_flow",
        "details",
        "key_points",
        "decisions",
        "action_items",
        "risks",
        "issues",
        "open_questions",
        "next_steps",
        "speaker_summary",
    ]
    for field in list_fields:
        items = _flatten_items(summary_obj.get(field))
        for i, item in enumerate(items):
            add_doc(item, "summary", field, idx=i)

    chunks = summary_data.get("chunks", [])
    if isinstance(chunks, list):
        for c_idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                add_doc(_as_text(chunk), "chunk", "raw", chunk_idx=c_idx)
                continue
            chunk_summary = chunk.get("summary")
            if isinstance(chunk_summary, str):
                parsed = _parse_json_blob(chunk_summary)
                if isinstance(parsed, dict):
                    chunk_summary_obj = parsed
                else:
                    chunk_summary_obj = {"summary": chunk_summary}
            elif isinstance(chunk_summary, dict):
                chunk_summary_obj = chunk_summary
            else:
                chunk_summary_obj = {}

            add_doc(_as_text(chunk_summary_obj.get("summary")), "chunk", "summary", chunk_idx=c_idx)
            add_doc(_as_text(chunk_summary_obj.get("context")), "chunk", "context", chunk_idx=c_idx)
            for field in list_fields:
                items = _flatten_items(chunk_summary_obj.get(field))
                for i, item in enumerate(items):
                    add_doc(item, "chunk", field, idx=i, chunk_idx=c_idx)

    return docs


def _collect_qc_docs(qc_data: Dict[str, Any], origin: str) -> List[Tuple[str, Dict[str, Any]]]:
    docs: List[Tuple[str, Dict[str, Any]]] = []
    if not isinstance(qc_data, dict):
        return docs

    def add_doc(text: str, field: str, idx: Optional[int] = None) -> None:
        cleaned = _clean_text(text)
        if not cleaned:
            return
        meta = _prune_metadata({
            "origin": origin,
            "section": "qc",
            "field": field,
            "idx": idx,
            "created_at": qc_data.get("created_at") or utc_now_iso(),
        })
        docs.append((cleaned, meta))

    for field in ["verdict", "retry_suggested"]:
        if field in qc_data:
            add_doc(_as_text(qc_data.get(field)), field)

    for field in ["warnings", "errors", "action_item_issues", "action_items_fixed"]:
        for i, item in enumerate(_flatten_items(qc_data.get(field))):
            add_doc(item, field, idx=i)

    metrics = qc_data.get("metrics")
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            add_doc(f"{key}: {value}", "metrics")

    return docs


def _chunked(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _build_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: Optional[str],
) -> List[List[float]]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: sentence-transformers. Install via `pip install sentence-transformers`."
        ) from exc

    model = SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def _get_collection(db_path: Path, name: str, reset: bool):
    try:
        import chromadb
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: chromadb. Install via `pip install chromadb`."
        ) from exc

    client = chromadb.PersistentClient(path=str(db_path))
    if reset:
        try:
            client.delete_collection(name=name)
        except Exception:
            pass
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed summary.json + qc_report.json into ChromaDB."
    )
    parser.add_argument("origin", nargs="?", default=None)
    parser.add_argument("--summary-root", type=str, default=None)
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION)
    parser.add_argument("--model", type=str, default=EMBED_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_root = Path(args.summary_root) if args.summary_root else DEFAULT_SUMMARY_DIR
    db_path = Path(args.db_path) if args.db_path else DEFAULT_DB_DIR
    db_path.mkdir(parents=True, exist_ok=True)

    summary_path = load_summary_json(summary_root, args.origin)
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    origin_stem = origin_from_summary_filename(summary_path.name)

    qc_path = load_qc_json(summary_path)
    qc_data = json.loads(qc_path.read_text(encoding="utf-8")) if qc_path else None

    docs = _collect_summary_docs(summary_data, origin_stem)
    if qc_data:
        docs.extend(_collect_qc_docs(qc_data, origin_stem))

    if not docs:
        raise SystemExit("No documents found to embed.")

    texts = [t for t, _ in docs]
    metadatas = [m for _, m in docs]

    embeddings = _build_embeddings(texts, args.model, args.batch_size, args.device)

    ids = [
        f"{origin_stem}:{m['section']}:{m['field']}:{m.get('chunk_idx') if m.get('chunk_idx') is not None else 'na'}:{m.get('idx') if m.get('idx') is not None else 'na'}:{i}"
        for i, m in enumerate(metadatas)
    ]

    collection = _get_collection(db_path, args.collection, args.reset)

    for batch in _chunked(list(range(len(texts))), args.batch_size):
        batch_texts = [texts[i] for i in batch]
        batch_metas = [metadatas[i] for i in batch]
        batch_ids = [ids[i] for i in batch]
        batch_embs = [embeddings[i] for i in batch]
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metas,
            embeddings=batch_embs,
        )

    print(
        f"Embedded {len(texts)} docs into collection '{args.collection}' at {db_path}"
    )


if __name__ == "__main__":
    main()