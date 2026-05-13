#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embed_and_query_chroma_test.py

목적:
- 엑셀 1개를 ChromaDB 컬렉션 1개에 임베딩
- 검색 시:
    1) 텍스트(TF-IDF) 기반 top 50
    2) Chroma dense retrieval top 50
    3) 합쳐서 최대 100개
    4) BAAI/bge-reranker-v2-m3 로 rerank
    5) 상위 10개로 답변/출력
- 같은 스크립트를 터미널 5개에서 동시에 띄워
  서로 다른 엑셀 / 서로 다른 persist_dir / 서로 다른 collection 으로
  병렬 임베딩 테스트

사용 예:
python embed_and_query_chroma_test.py \
  --excel "/data/set_A/더미데이터_1_A.xlsx" \
  --persist_dir "/data/chroma/collection_A" \
  --collection_name "collection_A"

권장 디렉토리 예:
project/
  data/
    set_A/더미데이터_1_A.xlsx
    set_B/더미데이터_2_B.xlsx
    set_C/더미데이터_3_C.xlsx
    set_D/더미데이터_4_D.xlsx
    set_E/더미데이터_5_E.xlsx
  chroma/
    collection_A/
    collection_B/
    collection_C/
    collection_D/
    collection_E/
"""

import os
import re
import json
import time
import math
import argparse
from typing import List, Dict, Any, Tuple

import pandas as pd
import chromadb

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


EMBED_MODEL = "BAAI/bge-m3"
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"


def normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def build_document(row: pd.Series, source_file: str) -> str:
    product_name = normalize_text(row["제품명"])
    price = int(row["가격"])
    # 테스트용 문서 텍스트
    # 제품 코드 검색, 숫자 검색, 자연어 검색이 모두 되도록 약간 풍부하게 작성
    doc = (
        f"소스파일: {source_file}\n"
        f"제품명: {product_name}\n"
        f"가격: {price}원\n"
        f"이 문서는 상품 코드 {product_name} 의 가격 정보입니다.\n"
        f"{product_name} 상품의 등록 가격은 {price}원 입니다."
    )
    return doc


def load_excel_as_docs(excel_path: str) -> List[Dict[str, Any]]:
    df = pd.read_excel(excel_path)

    required_cols = ["제품명", "가격"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"엑셀에 '{col}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

    source_file = os.path.basename(excel_path)
    docs: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        product_name = normalize_text(row["제품명"])
        price = int(row["가격"])
        doc_text = build_document(row, source_file)
        doc_id = f"{os.path.splitext(source_file)[0]}__{product_name}"

        docs.append({
            "id": doc_id,
            "document": doc_text,
            "metadata": {
                "source_file": source_file,
                "row_index": int(idx) + 2,  # header 고려
                "product_name": product_name,
                "price": price,
            }
        })
    return docs


def chunked(lst: List[Any], size: int):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def ensure_collection(client: chromadb.PersistentClient, collection_name: str, rebuild: bool = False):
    if rebuild:
        try:
            client.delete_collection(collection_name)
            print(f"[INFO] 기존 컬렉션 삭제: {collection_name}")
        except Exception:
            pass

    collection = client.get_or_create_collection(name=collection_name)
    return collection


def embed_documents(embedder: SentenceTransformer, texts: List[str], batch_size: int = 64) -> List[List[float]]:
    embs = embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embs.tolist()


def index_to_chroma(
    excel_path: str,
    persist_dir: str,
    collection_name: str,
    rebuild: bool,
    embed_batch_size: int,
):
    os.makedirs(persist_dir, exist_ok=True)

    print(f"[INFO] 엑셀 로드: {excel_path}")
    docs = load_excel_as_docs(excel_path)
    print(f"[INFO] 문서 수: {len(docs)}")

    print(f"[INFO] 임베딩 모델 로드: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    print(f"[INFO] Chroma PersistentClient: {persist_dir}")
    client = chromadb.PersistentClient(path=persist_dir)
    collection = ensure_collection(client, collection_name, rebuild=rebuild)

    # 이미 문서가 있는지 체크
    existing_count = collection.count()
    if existing_count > 0 and not rebuild:
        print(f"[INFO] 기존 컬렉션에 이미 데이터 존재: {existing_count}건")
        return embedder, client, collection, docs

    all_texts = [d["document"] for d in docs]
    all_ids = [d["id"] for d in docs]
    all_metas = [d["metadata"] for d in docs]

    print("[INFO] 전체 문서 임베딩 시작")
    all_embeddings = embed_documents(embedder, all_texts, batch_size=embed_batch_size)

    print("[INFO] Chroma add 시작")
    batch_size = 256
    for batch_idx, batch in enumerate(chunked(list(range(len(docs))), batch_size), start=1):
        ids = [all_ids[i] for i in batch]
        documents = [all_texts[i] for i in batch]
        metadatas = [all_metas[i] for i in batch]
        embeddings = [all_embeddings[i] for i in batch]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        print(f"[INFO] add batch {batch_idx}: {len(ids)}건")

    print(f"[INFO] 인덱싱 완료. collection.count() = {collection.count()}")
    return embedder, client, collection, docs


class HybridRetriever:
    def __init__(
        self,
        docs: List[Dict[str, Any]],
        collection,
        embedder: SentenceTransformer,
        reranker: CrossEncoder,
        tfidf_ngram=(2, 4),
    ):
        self.docs = docs
        self.collection = collection
        self.embedder = embedder
        self.reranker = reranker

        self.doc_texts = [d["document"] for d in docs]
        self.doc_map = {d["id"]: d for d in docs}

        # 제품코드 검색이 잘 되도록 char_wb 사용
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=tfidf_ngram)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)

    def text_retrieve(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        qv = self.vectorizer.transform([query])
        sims = linear_kernel(qv, self.tfidf_matrix).flatten()

        top_idx = sims.argsort()[::-1][:top_k]
        results = []
        for rank, i in enumerate(top_idx, start=1):
            d = self.docs[i]
            results.append({
                "id": d["id"],
                "document": d["document"],
                "metadata": d["metadata"],
                "retrieval_score": float(sims[i]),
                "retrieval_source": "text_tfidf",
                "rank_in_source": rank,
            })
        return results

    def dense_retrieve(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        q_emb = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        res = self.collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        ids = res["ids"][0]
        documents = res["documents"][0]
        metadatas = res["metadatas"][0]
        distances = res["distances"][0]

        for rank, (doc_id, doc_text, meta, dist) in enumerate(zip(ids, documents, metadatas, distances), start=1):
            # distance는 작을수록 가까움인 경우가 많으므로 보기 좋게 score 변환
            score = 1.0 / (1.0 + float(dist))
            results.append({
                "id": doc_id,
                "document": doc_text,
                "metadata": meta,
                "retrieval_score": score,
                "retrieval_source": "dense_chroma",
                "rank_in_source": rank,
            })
        return results

    def hybrid_retrieve_and_rerank(
        self,
        query: str,
        text_top_k: int = 50,
        dense_top_k: int = 50,
        final_top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        text_hits = self.text_retrieve(query, top_k=text_top_k)
        dense_hits = self.dense_retrieve(query, top_k=dense_top_k)

        merged: Dict[str, Dict[str, Any]] = {}

        for hit in text_hits + dense_hits:
            if hit["id"] not in merged:
                merged[hit["id"]] = {
                    "id": hit["id"],
                    "document": hit["document"],
                    "metadata": hit["metadata"],
                    "sources": [hit["retrieval_source"]],
                    "text_score": hit["retrieval_score"] if hit["retrieval_source"] == "text_tfidf" else None,
                    "dense_score": hit["retrieval_score"] if hit["retrieval_source"] == "dense_chroma" else None,
                }
            else:
                merged[hit["id"]]["sources"].append(hit["retrieval_source"])
                if hit["retrieval_source"] == "text_tfidf":
                    merged[hit["id"]]["text_score"] = hit["retrieval_score"]
                if hit["retrieval_source"] == "dense_chroma":
                    merged[hit["id"]]["dense_score"] = hit["retrieval_score"]

        candidates = list(merged.values())
        print(f"[INFO] text {len(text_hits)} + dense {len(dense_hits)} -> dedupe {len(candidates)}")

        pairs = [[query, c["document"]] for c in candidates]
        rerank_scores = self.reranker.predict(pairs, batch_size=16, show_progress_bar=False)

        for c, score in zip(candidates, rerank_scores):
            c["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:final_top_k]


def make_simple_answer(query: str, top_docs: List[Dict[str, Any]]) -> str:
    if not top_docs:
        return "관련 문서를 찾지 못했습니다."

    lines = []
    lines.append("=== 최종 상위 10개 문서 기준 답변 ===")
    lines.append(f"질문: {query}")
    lines.append("")

    # 간단한 테스트용 요약
    prices = []
    for d in top_docs:
        meta = d["metadata"]
        product_name = meta.get("product_name")
        price = meta.get("price")
        if product_name is not None and price is not None:
            prices.append((product_name, int(price)))

    if prices:
        min_item = min(prices, key=lambda x: x[1])
        max_item = max(prices, key=lambda x: x[1])
        lines.append(f"- 상위 문서 내 최저가 상품: {min_item[0]} / {min_item[1]}원")
        lines.append(f"- 상위 문서 내 최고가 상품: {max_item[0]} / {max_item[1]}원")
        lines.append("")

    lines.append("상위 문서:")
    for i, d in enumerate(top_docs, start=1):
        meta = d["metadata"]
        lines.append(
            f"{i:02d}. "
            f"product={meta.get('product_name')} | "
            f"price={meta.get('price')} | "
            f"file={meta.get('source_file')} | "
            f"rerank={d.get('rerank_score'):.6f} | "
            f"sources={','.join(sorted(set(d.get('sources', []))))}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", required=True, help="임베딩할 엑셀 파일 경로")
    parser.add_argument("--persist_dir", required=True, help="Chroma persist 디렉토리")
    parser.add_argument("--collection_name", required=True, help="Chroma collection 이름")
    parser.add_argument("--rebuild", action="store_true", help="기존 컬렉션 삭제 후 재구축")
    parser.add_argument("--embed_batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None, help="예: cpu, cuda")
    args = parser.parse_args()

    start = time.time()

    print("=" * 80)
    print("[CONFIG]")
    print(json.dumps({
        "excel": args.excel,
        "persist_dir": args.persist_dir,
        "collection_name": args.collection_name,
        "rebuild": args.rebuild,
        "embed_model": EMBED_MODEL,
        "reranker": RERANKER_NAME,
    }, ensure_ascii=False, indent=2))
    print("=" * 80)

    # 임베더/리랭커 로드
    if args.device:
        print(f"[INFO] 임베딩 모델 로드(device={args.device}): {EMBED_MODEL}")
        embedder = SentenceTransformer(EMBED_MODEL, device=args.device)
        print(f"[INFO] 리랭커 로드(device={args.device}): {RERANKER_NAME}")
        reranker = CrossEncoder(RERANKER_NAME, device=args.device)
    else:
        print(f"[INFO] 임베딩 모델 로드: {EMBED_MODEL}")
        embedder = SentenceTransformer(EMBED_MODEL)
        print(f"[INFO] 리랭커 로드: {RERANKER_NAME}")
        reranker = CrossEncoder(RERANKER_NAME)

    # 인덱싱
    os.makedirs(args.persist_dir, exist_ok=True)
    docs = load_excel_as_docs(args.excel)

    client = chromadb.PersistentClient(path=args.persist_dir)
    collection = ensure_collection(client, args.collection_name, rebuild=args.rebuild)

    if collection.count() == 0 or args.rebuild:
        all_texts = [d["document"] for d in docs]
        all_ids = [d["id"] for d in docs]
        all_metas = [d["metadata"] for d in docs]

        print("[INFO] 임베딩 생성 시작")
        all_embeddings = embedder.encode(
            all_texts,
            batch_size=args.embed_batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).tolist()

        batch_size = 256
        for batch_idx, batch in enumerate(chunked(list(range(len(docs))), batch_size), start=1):
            ids = [all_ids[i] for i in batch]
            documents = [all_texts[i] for i in batch]
            metadatas = [all_metas[i] for i in batch]
            embeddings = [all_embeddings[i] for i in batch]

            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            print(f"[INFO] Chroma add batch {batch_idx}: {len(ids)}건")

    print(f"[INFO] collection.count() = {collection.count()}")
    print(f"[INFO] 인덱싱 완료. elapsed={time.time() - start:.2f}s")

    retriever = HybridRetriever(
        docs=docs,
        collection=collection,
        embedder=embedder,
        reranker=reranker,
    )

    print("\n[READY] 질의를 입력하세요. 종료하려면 exit")
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

        t0 = time.time()
        top_docs = retriever.hybrid_retrieve_and_rerank(
            query=query,
            text_top_k=50,
            dense_top_k=50,
            final_top_k=10,
        )
        answer = make_simple_answer(query, top_docs)
        print(answer)
        print(f"[INFO] query elapsed = {time.time() - t0:.3f}s")


if __name__ == "__main__":
    main()