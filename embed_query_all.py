#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embed_query_all.py
- embed_test/chroma 아래 여러 PersistentClient(DB)에 나뉜 컬렉션 동시 검색
- dense 검색 후 리랭커로 합쳐서 출력
- embed_docs.py가 넣는 metadata(filename, source_path, page, section_type, page_kind 등) 활용
- 질의 결과 sources 항목에 source_pdf(store, file, page)·page_anchor·(API에서 source_pdf_url 보강) 제공
- OCR/visual_summary/body_text/retrieval_text 가중치 반영
- torch / device 상태를 조금 더 명확히 출력
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

EMBED_MODEL = "BAAI/bge-m3"
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
DOC_EXCERPT_CHARS = 900
DEFAULT_LOCAL_LLM_MODEL = "/home/siwasoft/gpt-oss-20b-GGUF/gpt-oss-20b-Q2_K_L.gguf"
DEFAULT_LLM_N_CTX = 8192
DEFAULT_LLM_MAX_TOKENS = 1024

# with_answer=True 인데 LLM 단계가 실패했을 때 사용자에게 줄 짧은 안내(검색 발췌 전체는 넣지 않음)
_LLM_ANSWER_FAILED_MSG = (
    "답변 생성에 실패했습니다. 검색 근거는 응답의 retrieval_answer·sources 필드를 참고해 주세요."
)

DEFAULT_CHROMA_BASE = Path(__file__).resolve().parent / "embed_test" / "chroma"
_LOCAL_LLM_CACHE: Dict[str, Any] = {}
_INTERNAL_TAG_RE = re.compile(r"<\|[^|>]+\|>")


def discover_stores(chroma_base: Path) -> List[Tuple[str, Path]]:
    if not chroma_base.is_dir():
        return []
    out: List[Tuple[str, Path]] = []
    for child in sorted(chroma_base.iterdir()):
        if child.is_dir() and (child / "chroma.sqlite3").exists():
            out.append((child.name, child))
    return out


def list_collection_names(client: chromadb.PersistentClient) -> List[str]:
    cols = client.list_collections()
    return sorted([c.name for c in cols])


def pick_device(cli_device: Optional[str]) -> str:
    if cli_device:
        return cli_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def dense_hits_from_store(
    label: str,
    persist_dir: Path,
    collection_name: str,
    embedder: SentenceTransformer,
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(name=collection_name)

    q_emb = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).tolist()

    res = collection.query(
        query_embeddings=q_emb,
        n_results=min(top_k, max(1, collection.count())),
        include=["documents", "metadatas", "distances"],
    )

    hits: List[Dict[str, Any]] = []
    ids = res["ids"][0]
    documents = res["documents"][0]
    metadatas = res["metadatas"][0]
    distances = res["distances"][0]

    for rank, (doc_id, doc_text, meta, dist) in enumerate(
        zip(ids, documents, metadatas, distances),
        start=1,
    ):
        score = 1.0 / (1.0 + float(dist))
        meta = dict(meta or {})
        meta["chroma_store"] = label
        meta["chroma_collection"] = collection_name
        hits.append(
            {
                "uniq_key": f"{label}::{collection_name}::{doc_id}",
                "id": doc_id,
                "document": doc_text,
                "metadata": meta,
                "retrieval_score": score,
                "rank_in_store": rank,
            }
        )
    return hits


def _meta_source_label(meta: Dict[str, Any]) -> str:
    if meta.get("original_filename"):
        return str(meta["original_filename"])
    if meta.get("source_file"):
        return str(meta["source_file"])
    if meta.get("filename"):
        return str(meta["filename"])
    if meta.get("file_name"):
        return str(meta["file_name"])
    sp = meta.get("source_path") or meta.get("file_path")
    if sp:
        return Path(str(sp)).name
    return ""


def _meta_page(meta: Dict[str, Any]) -> Optional[int]:
    page = meta.get("page")
    if page is None:
        page = meta.get("page_num")
    try:
        return int(page) if page is not None else None
    except Exception:
        return None


def _meta_section_type(meta: Dict[str, Any]) -> str:
    raw = meta.get("section_type") or ""
    if raw == "body":
        return "body_text"
    return str(raw) if raw else "unknown"


def _excerpt(text: Optional[str], max_chars: int) -> str:
    if not text:
        return ""
    t = str(text).strip().replace("\r\n", "\n")
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3] + "..."


def is_visual_query(query: str) -> bool:
    q = query.replace(" ", "")
    patterns = [
        r"빨간박스", r"빨간", r"강조", r"표시", r"필수", r"입력값", r"필수입력", r"어디", r"어느칸",
        r"무슨칸", r"화면", r"보이는", r"번호", r"단계", r"클릭", r"스캔", r"OCR", r"이미지",
        r"캡처", r"입력", r"양식", r"폼",
    ]
    return any(re.search(p, q, flags=re.I) for p in patterns)


def compute_section_bonus(query: str, meta: Dict[str, Any]) -> float:
    st = _meta_section_type(meta)
    bonus = 0.0
    visual_q = is_visual_query(query)

    if visual_q:
        if st == "retrieval_text":
            bonus += 0.14
        elif st == "visual_summary":
            bonus += 0.12
        elif st == "ocr_text":
            bonus += 0.08
        elif st == "body_text":
            bonus += 0.03
    else:
        if st == "body_text":
            bonus += 0.10
        elif st == "retrieval_text":
            bonus += 0.06
        elif st == "visual_summary":
            bonus += 0.03

    if meta.get("has_red_box"):
        bonus += 0.03 if visual_q else 0.01
    if meta.get("has_step_numbers"):
        bonus += 0.02 if visual_q else 0.00
    if meta.get("page_kind") == "manual_ui":
        bonus += 0.03 if visual_q else 0.01
    if meta.get("erp_detected") or meta.get("has_erp_screen"):
        bonus += 0.03 if visual_q else 0.01
    return bonus


def merge_and_rerank(
    all_hits: List[Dict[str, Any]],
    query: str,
    reranker: CrossEncoder,
    final_k: int,
) -> List[Dict[str, Any]]:
    if not all_hits:
        return []

    pairs = [[query, h["document"]] for h in all_hits]
    scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)

    for h, s in zip(all_hits, scores):
        meta = h.get("metadata") or {}
        rerank_score = float(s)
        bonus = compute_section_bonus(query, meta)
        h["rerank_score"] = rerank_score
        h["section_bonus"] = bonus
        h["final_score"] = rerank_score + bonus

    all_hits.sort(key=lambda x: x["final_score"], reverse=True)
    return all_hits[:final_k]


def make_answer(query: str, top_docs: List[Dict[str, Any]], excerpt_chars: int = DOC_EXCERPT_CHARS) -> str:
    if not top_docs:
        return "관련 문서를 찾지 못했습니다."

    lines: List[str] = []
    lines.append("=== 전체 컬렉션 통합 검색 (dense + rerank) ===")
    lines.append(f"질문: {query}")
    lines.append(f"시각질문 보정: {'ON' if is_visual_query(query) else 'OFF'}")
    lines.append("")
    lines.append("상위 근거 청크 (본문 일부):")

    for i, d in enumerate(top_docs, start=1):
        meta = d.get("metadata") or {}
        store = meta.get("chroma_store", "")
        src = _meta_source_label(meta)
        page = _meta_page(meta)
        section_type = _meta_section_type(meta)
        page_kind = meta.get("page_kind")
        loc_bits: List[str] = []
        if page is not None:
            loc_bits.append(f"page={page}")
        if section_type:
            loc_bits.append(f"section={section_type}")
        if page_kind:
            loc_bits.append(f"kind={page_kind}")
        loc = f" ({', '.join(loc_bits)})" if loc_bits else ""

        doc = d.get("document") or ""
        body = _excerpt(doc, excerpt_chars)

        lines.append(
            f"\n--- [{i:02d}] store={store} | final={d.get('final_score', 0):.6f} | "
            f"rerank={d.get('rerank_score', 0):.6f} | bonus={d.get('section_bonus', 0):.3f} | "
            f"file={src or '(unknown)'}{loc} ---"
        )
        lines.append(body if body else "(문서 본문 없음)")

    lines.append("")
    lines.append("※ 위는 검색된 원문 발췌입니다. 요약 답변은 LLM 단계를 추가하세요.")
    return "\n".join(lines)


def resolve_collection_name(persist_dir: Path, explicit: Optional[str]) -> Optional[str]:
    client = chromadb.PersistentClient(path=str(persist_dir))
    names = list_collection_names(client)
    if not names:
        return None
    if explicit:
        # 권한/정책으로 특정 컬렉션이 강제된 경우에는 fallback 하지 않는다.
        # (없으면 이 스토어를 건너뛰어야 다른 컬렉션으로 새지 않음)
        return explicit if explicit in names else None
    guess = persist_dir.name
    if guess in names:
        return guess
    if len(names) == 1:
        return names[0]
    return names[0]


def _meta_for_json(meta: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        out[str(k)] = str(v)
    return out


def _meta_pdf_basename(meta: Dict[str, Any]) -> str:
    fp = meta.get("file_path") or meta.get("source_path") or ""
    if fp:
        return Path(str(fp)).name
    return str(meta.get("filename") or meta.get("file_name") or "")


def _top_doc_for_json(d: Dict[str, Any], excerpt_chars: int) -> Dict[str, Any]:
    meta = d.get("metadata") or {}
    page = _meta_page(meta)
    basename = (_meta_pdf_basename(meta) or "").strip()
    store = str(meta.get("chroma_store") or "").strip()
    source_pdf: Optional[Dict[str, Any]] = None
    page_anchor: Optional[str] = None
    if basename and store:
        source_pdf = {"store": store, "file": basename, "page": page}
        if page is not None:
            page_anchor = f"#page={int(page)}"

    return {
        "id": d.get("id"),
        "chroma_store": meta.get("chroma_store", ""),
        "chroma_collection": meta.get("chroma_collection", ""),
        "source_file": _meta_source_label(meta),
        "page": page,
        "section_type": _meta_section_type(meta),
        "final_score": float(d.get("final_score", 0.0) or 0.0),
        "rerank_score": float(d.get("rerank_score", 0.0) or 0.0),
        "section_bonus": float(d.get("section_bonus", 0.0) or 0.0),
        "document_excerpt": _excerpt(d.get("document"), excerpt_chars),
        "metadata": _meta_for_json(meta),
        "source_pdf": source_pdf,
        "page_anchor": page_anchor,
    }


def _source_label_for_citation(meta: Dict[str, Any]) -> str:
    src = _meta_source_label(meta) or "(unknown)"
    page = _meta_page(meta)
    if page is not None:
        return f"{src} p.{page}"
    return src


def _build_llm_evidence(top_docs: List[Dict[str, Any]], excerpt_chars: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for i, d in enumerate(top_docs, start=1):
        meta = d.get("metadata") or {}
        items.append(
            {
                "rank": i,
                "source_label": _source_label_for_citation(meta),
                "score": round(float(d.get("final_score", 0.0) or 0.0), 6),
                "section_type": _meta_section_type(meta),
                "excerpt": _excerpt(d.get("document"), excerpt_chars),
            }
        )
    return items


def _build_llm_messages(query: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    system_prompt = (
        "너는 한국어로 답변하는 숙련된 리서치 어시스턴트다. "
        "아래 evidence만 근거로 답하라. 근거가 없으면 반드시 "
        "'제공된 자료에서 확인되지 않습니다.'라고 답하라.\n"
        "목록 서식(획일): 큰 절은 '1. 제목' '2. 제목'처럼 번호만 쓴다. "
        "각 절의 하위 단계는 전부 마크다운 불릿 '- ' 한 단계만 쓴다. "
        "'1) 2)' 같은 괄호 번호 목록은 쓰지 말고, 필요하면 불릿으로 나열한다.\n"
        "인용(획일): 근거가 있는 문단·불릿 묶음 끝에는 반드시 "
        "단 한 줄로 '[출처: source_label]'만 쓴다. "
        "그 줄은 본문 문장과 같은 줄에 두지 말고, 본문이 끝난 직후 바로 다음 줄에만 둔다. "
        "출처 줄 다음에 새 번호 절(예: '2. ')이 시작되면 가독성을 위해 빈 줄 두 줄을 넣어 구분하라. "
        "문장 끝에 파일명·p.N만 덧붙이거나, p.N 뒤에 따옴표를 붙이지 마라. "
        "반드시 대괄호로 시작해 '출처:' 다음에 한 칸 뒤 evidence의 source_label 문자열을 "
        "그대로 붙이고 마지막은 ']'로 닫는다. "
        "']'만 단독으로 두거나 '출처:' 없이 파일명만 쓰는 것은 금지다. "
        "source_label은 evidence의 문자열과 동일하게 쓰되, 추가 따옴표로 감싸지 마라.\n"
        "마크다운 표를 쓸 경우 표 셀 안에는 [출처: ...]를 넣지 말고, 표 직후 문단이나 표 아래에만 인용하라(파이프(|)와 충돌 방지). "
        "'원본 PDF' 섹션, 출처 링크 목록, URL 목록은 절대 작성하지 마라."
    )
    payload = {
        "query": query,
        "evidence_count": len(evidence),
        "evidence": evidence,
    }
    user_prompt = (
        f"질의:\n{query}\n\n"
        f"[EVIDENCE_JSON_BEGIN]\n{json.dumps(payload, ensure_ascii=False)}\n[EVIDENCE_JSON_END]\n\n"
        "규칙: 1) evidence 밖 내용 추정 금지 2) 불확실하면 확인 불가 명시\n"
        "3) 인용은 오직 '[출처: source_label]' 전체 한 덩어리(앞뒤 대괄호 포함), "
        "label은 evidence와 동일, 따옴표 없음\n"
        "4) '원본 PDF' 제목/불릿/URL 링크는 작성 금지\n"
        "5) 표 사용 시 셀 내부 인용 금지, 표 밖에만 [출처: ...]\n"
        "6) 하위 단계는 '- '만 사용, '1) 2)' 형식 금지\n"
        "7) '[출처: …]'는 항상 단독 줄(본문 바로 다음 줄), 본문과 한 줄에 섞지 않기. "
        "새 번호 절 시작 전에는 출처 뒤에 빈 줄 두 줄 유지"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def _strip_internal_blocks(text: str) -> str:
    t = str(text or "").strip()
    if not t:
        return t

    # Some chat-formatted models emit internal channel blocks.
    # Keep only the part after the last final-message marker when present.
    final_marker = "<|channel|>final<|message|>"
    idx = t.rfind(final_marker)
    if idx != -1:
        t = t[idx + len(final_marker):].strip()

    # Remove remaining tool/chat control tags like <|start|>, <|end|>, <|channel|>, <|message|>.
    t = _INTERNAL_TAG_RE.sub("", t)

    # 선행 "analysis" 누설: "analysis We …", "analysisWe …"(공백 없음), "analysis: …" 등.
    # 몇 번 반복해 "analysis analysis …" 형태도 제거한다.
    for _ in range(6):
        prev = t
        t = re.sub(r"(?is)^analysis\s*[:\.]?\s*", "", t).lstrip()
        if t == prev:
            break

    # Normalize excessive blank lines.
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def _load_local_llm(model_path: str, n_ctx: int):
    cached = _LOCAL_LLM_CACHE.get(model_path)
    if cached is not None:
        return cached
    try:
        from llama_cpp import Llama
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "llama-cpp-python is required for local LLM mode. "
            "Install it first (pip install llama-cpp-python)."
        ) from e
    llm = Llama(
        model_path=model_path,
        n_ctx=max(2048, int(n_ctx)),
        n_gpu_layers=-1,
        verbose=False,
    )
    _LOCAL_LLM_CACHE[model_path] = llm
    return llm


def answer_with_local_llm(
    query: str,
    top_docs: List[Dict[str, Any]],
    *,
    model_path: str,
    excerpt_chars: int,
    n_ctx: int,
    max_tokens: int,
) -> str:
    if not top_docs:
        return "관련 문서를 찾지 못했습니다."
    evidence = _build_llm_evidence(top_docs, excerpt_chars=excerpt_chars)
    messages = _build_llm_messages(query, evidence)
    llm = _load_local_llm(model_path=model_path, n_ctx=n_ctx)
    out = llm.create_chat_completion(
        messages=messages,
        temperature=0.2,
        top_p=0.95,
        repeat_penalty=1.1,
        max_tokens=max(128, int(max_tokens)),
    )
    raw = str(((out.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    ans = _strip_internal_blocks(raw)
    return ans or "제공된 자료에서 확인되지 않습니다."


def run_single_query(
    query: str,
    *,
    chroma_base: Path,
    embedder: SentenceTransformer,
    reranker: CrossEncoder,
    per_collection_k: int = 20,
    final_k: int = 10,
    excerpt_chars: int = DOC_EXCERPT_CHARS,
    collection_name: Optional[str] = None,
    with_answer: bool = False,
    model_id: Optional[str] = None,
    llm_n_ctx: int = DEFAULT_LLM_N_CTX,
    llm_max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
) -> Dict[str, Any]:
    """
    터미널 `Q>` 루프 없이 한 번의 질의만 실행. HTTP API 등에서 사용.
    """
    q = (query or "").strip()
    if not q:
        return {"ok": False, "error": "empty_query", "message": "질문이 비어 있습니다."}

    chroma_base = chroma_base.resolve()
    stores = discover_stores(chroma_base)
    if not stores:
        return {
            "ok": False,
            "error": "no_chroma_stores",
            "message": f"Chroma 스토어를 찾지 못했습니다: {chroma_base}",
        }

    t0 = time.time()
    all_hits: List[Dict[str, Any]] = []
    store_errors: List[Dict[str, str]] = []

    for label, persist_dir in stores:
        cname = resolve_collection_name(persist_dir, collection_name)
        if not cname:
            store_errors.append({"store": label, "error": "no_collection"})
            continue
        try:
            hits = dense_hits_from_store(
                label=label,
                persist_dir=persist_dir,
                collection_name=cname,
                embedder=embedder,
                query=q,
                top_k=per_collection_k,
            )
            all_hits.extend(hits)
        except Exception as e:
            store_errors.append({"store": label, "error": str(e)})

    top_docs = merge_and_rerank(all_hits, q, reranker, final_k)
    retrieval_answer = make_answer(q, top_docs, excerpt_chars=excerpt_chars)
    answer = retrieval_answer
    llm_error: Optional[str] = None
    selected_model = (model_id or "").strip() or DEFAULT_LOCAL_LLM_MODEL
    if with_answer:
        model_path = selected_model
        if selected_model.startswith("local:"):
            model_path = selected_model[len("local:"):].strip() or DEFAULT_LOCAL_LLM_MODEL
        try:
            answer = answer_with_local_llm(
                q,
                top_docs,
                model_path=model_path,
                excerpt_chars=excerpt_chars,
                n_ctx=llm_n_ctx,
                max_tokens=llm_max_tokens,
            )
        except Exception as e:
            llm_error = str(e)
            answer = _LLM_ANSWER_FAILED_MSG
    elapsed = time.time() - t0

    out = {
        "ok": True,
        "query": q,
        "chroma_base": str(chroma_base),
        "visual_query": is_visual_query(q),
        "num_candidates": len(all_hits),
        "num_final": len(top_docs),
        "elapsed_sec": round(elapsed, 4),
        "answer": answer,
        "retrieval_answer": retrieval_answer,
        "sources": [_top_doc_for_json(d, excerpt_chars) for d in top_docs],
        "store_errors": store_errors,
        "with_answer": bool(with_answer),
        "model_id": selected_model,
    }
    if llm_error:
        out["llm_error"] = llm_error
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="여러 Chroma 스토어 전체 dense 검색 + 리랭크")
    parser.add_argument("--chroma_base", type=str, default=str(DEFAULT_CHROMA_BASE), help="chroma.sqlite3가 들어 있는 상위 폴더")
    parser.add_argument("--per_collection_k", type=int, default=20, help="스토어당 가져올 상위 개수")
    parser.add_argument("--final_k", type=int, default=10, help="리랭크 후 최종 개수")
    parser.add_argument("--device", type=str, default=None, help="예: cuda, cpu")
    parser.add_argument("--embed_model", type=str, default=EMBED_MODEL, help="질의 임베딩 모델")
    parser.add_argument("--reranker", type=str, default=RERANKER_NAME, help="리랭커 모델")
    parser.add_argument("--excerpt_chars", type=int, default=DOC_EXCERPT_CHARS, help="출력에 보여줄 청크 본문 최대 글자 수")
    parser.add_argument("--collection", type=str, default=None, help="각 스토어에서 사용할 컬렉션명(미지정 시 자동 선택)")
    args = parser.parse_args()

    chroma_base = Path(args.chroma_base).resolve()
    stores = discover_stores(chroma_base)
    if not stores:
        print(f"[ERROR] Chroma 스토어를 찾지 못했습니다: {chroma_base}")
        print("  (각 DB는 .../collection_X/chroma.sqlite3 구조여야 합니다.)")
        return

    print("[INFO] 검색 대상 스토어:")
    for label, p in stores:
        print(f"  - {label}  ({p})")

    device = pick_device(args.device)
    print(f"[INFO] torch: {torch.__version__} | cuda={torch.cuda.is_available()} | device={device}")

    embed_id = args.embed_model
    reranker_id = args.reranker

    print(f"[INFO] 임베딩 로드 device={device}: {embed_id}")
    embedder = SentenceTransformer(embed_id, device=device)
    print(f"[INFO] 리랭커 로드 device={device}: {reranker_id}")
    reranker = CrossEncoder(reranker_id, device=device)

    print("\n[READY] 질문 입력 (exit 로 종료)\n")

    while True:
        try:
            query = input("\nQ> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] 종료")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("[INFO] 종료")
            break

        out = run_single_query(
            query,
            chroma_base=chroma_base,
            embedder=embedder,
            reranker=reranker,
            per_collection_k=args.per_collection_k,
            final_k=args.final_k,
            excerpt_chars=args.excerpt_chars,
            collection_name=args.collection,
        )
        if not out.get("ok"):
            print(f"[ERROR] {out.get('message', out)}")
            continue
        print(out["answer"])
        print(
            f"[INFO] 후보 {out['num_candidates']}건 → 최종 {out['num_final']}건, "
            f"elapsed={out['elapsed_sec']:.3f}s"
        )
        for err in out.get("store_errors") or []:
            if err.get("error") != "no_collection":
                print(f"[WARN] store={err.get('store')}: {err.get('error')}")


if __name__ == "__main__":
    main()
