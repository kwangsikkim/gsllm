import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import chromadb
from FlagEmbedding import BGEM3FlagModel


BASE_DIR = Path("/home/siwasoft/gsllm")
CHROMA_DIR = BASE_DIR / "chroma_data"
DUCKDB_PATH = BASE_DIR / "sales.duckdb"

ROW_COLLECTION = "sales_row_v1"
AGG_COLLECTION = "sales_agg_v1"
SCHEMA_COLLECTION = "sales_schema_v1"

EMBED_MODEL = "BAAI/bge-m3"
USE_FP16 = True

KNOWN_PRODUCT_TYPES = ["과자", "음료", "빙과", "식품", "기타", "미분류"]
KNOWN_ORG_TYPES = ["유통", "직거래", "특판", "온라인", "대리점", "수출", "미분류"]


class QueryEmbedder:
    def __init__(self, model_name: str, use_fp16: bool = True, max_length: int = 256):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.max_length = max_length

    def encode_query(self, text: str):
        out = self.model.encode(
            [text],
            batch_size=1,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return out["dense_vecs"][0]


def extract_dates(text: str) -> List[str]:
    return re.findall(r"\b(20\d{6})\b", text)


def extract_known_value(text: str, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in text:
            return c
    return None


def extract_explicit_org_type(text: str) -> Optional[str]:
    patterns = [
        r"조직구분\s*[:=]\s*([가-힣A-Za-z0-9_]+)",
        r"\borg\s*[:=]\s*([가-힣A-Za-z0-9_]+)",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            value = m.group(1).strip()
            if value in KNOWN_ORG_TYPES:
                return value
    return None


def build_where_filter(query: str) -> Optional[Dict]:
    dates = extract_dates(query)
    product_type = extract_known_value(query, KNOWN_PRODUCT_TYPES)
    org_type = extract_explicit_org_type(query)

    conditions = []

    if len(dates) == 1:
        conditions.append({"일자": {"$eq": dates[0]}})
    if product_type:
        conditions.append({"제품구분": {"$eq": product_type}})
    if org_type:
        conditions.append({"조직구분": {"$eq": org_type}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def get_chroma_collection(name: str):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(name)


def search_collection(
    collection_name: str,
    query_embedding,
    where: Optional[Dict] = None,
    n_results: int = 5,
):
    col = get_chroma_collection(collection_name)
    result = col.query(
        query_embeddings=[query_embedding],
        where=where,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    return result


def pretty_print_chroma_result(title: str, result: Dict):
    print(f"\n===== {title} =====")

    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0] if "distances" in result else []

    if not ids:
        print("검색 결과 없음")
        return

    for i, doc_id in enumerate(ids, start=1):
        print(f"\n[{i}] id = {doc_id}")
        if i - 1 < len(dists):
            print(f"distance = {dists[i - 1]}")
        if i - 1 < len(metas):
            print("metadata =", json.dumps(metas[i - 1], ensure_ascii=False))
        if i - 1 < len(docs):
            print("document =", docs[i - 1])


def connect_duckdb():
    return duckdb.connect(str(DUCKDB_PATH), read_only=True)


def build_sql_filters_from_query(query: str) -> Tuple[str, List]:
    dates = extract_dates(query)
    product_type = extract_known_value(query, KNOWN_PRODUCT_TYPES)
    org_type = extract_explicit_org_type(query)

    clauses = []
    params = []

    if len(dates) == 1:
        clauses.append("일자 = ?")
        params.append(dates[0])
    elif len(dates) >= 2:
        clauses.append("일자 BETWEEN ? AND ?")
        params.extend([dates[0], dates[1]])

    if product_type:
        clauses.append("제품구분 = ?")
        params.append(product_type)

    if org_type:
        clauses.append("조직구분 = ?")
        params.append(org_type)

    where_sql = ""
    if clauses:
        where_sql = "WHERE " + " AND ".join(clauses)

    return where_sql, params


def duckdb_summary(query: str):
    con = connect_duckdb()
    try:
        where_sql, params = build_sql_filters_from_query(query)

        sql = f"""
        SELECT
            COUNT(*) AS row_count,
            SUM(판매수량) AS 판매수량합계,
            SUM(반품수량) AS 반품수량합계,
            SUM(총매출금액) AS 총매출금액합계,
            SUM(관리순매출금액) AS 관리순매출금액합계,
            SUM(할인금액) AS 할인금액합계,
            SUM(반품금액) AS 반품금액합계
        FROM raw_sales
        {where_sql}
        """
        row = con.execute(sql, params).fetchone()

        return {
            "row_count": row[0] or 0,
            "판매수량합계": row[1] or 0,
            "반품수량합계": row[2] or 0,
            "총매출금액합계": row[3] or 0,
            "관리순매출금액합계": row[4] or 0,
            "할인금액합계": row[5] or 0,
            "반품금액합계": row[6] or 0,
        }
    finally:
        con.close()


def duckdb_groupby_product(query: str, limit: int = 10):
    con = connect_duckdb()
    try:
        where_sql, params = build_sql_filters_from_query(query)

        sql = f"""
        SELECT
            제품구분,
            SUM(관리순매출금액) AS 관리순매출금액합계,
            SUM(총매출금액) AS 총매출금액합계,
            SUM(반품금액) AS 반품금액합계,
            SUM(할인금액) AS 할인금액합계
        FROM raw_sales
        {where_sql}
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT {limit}
        """
        rows = con.execute(sql, params).fetchall()

        return [
            {
                "제품구분": r[0],
                "관리순매출금액합계": r[1] or 0,
                "총매출금액합계": r[2] or 0,
                "반품금액합계": r[3] or 0,
                "할인금액합계": r[4] or 0,
            }
            for r in rows
        ]
    finally:
        con.close()


def duckdb_groupby_org(query: str, limit: int = 10):
    con = connect_duckdb()
    try:
        where_sql, params = build_sql_filters_from_query(query)

        sql = f"""
        SELECT
            조직구분,
            SUM(관리순매출금액) AS 관리순매출금액합계,
            SUM(총매출금액) AS 총매출금액합계,
            SUM(반품금액) AS 반품금액합계,
            SUM(할인금액) AS 할인금액합계
        FROM raw_sales
        {where_sql}
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT {limit}
        """
        rows = con.execute(sql, params).fetchall()

        return [
            {
                "조직구분": r[0],
                "관리순매출금액합계": r[1] or 0,
                "총매출금액합계": r[2] or 0,
                "반품금액합계": r[3] or 0,
                "할인금액합계": r[4] or 0,
            }
            for r in rows
        ]
    finally:
        con.close()


def duckdb_groupby_date(query: str, limit: int = 31):
    con = connect_duckdb()
    try:
        where_sql, params = build_sql_filters_from_query(query)

        sql = f"""
        SELECT
            일자,
            SUM(관리순매출금액) AS 관리순매출금액합계,
            SUM(총매출금액) AS 총매출금액합계,
            SUM(반품금액) AS 반품금액합계,
            SUM(할인금액) AS 할인금액합계
        FROM raw_sales
        {where_sql}
        GROUP BY 1
        ORDER BY 1
        LIMIT {limit}
        """
        rows = con.execute(sql, params).fetchall()

        return [
            {
                "일자": r[0],
                "관리순매출금액합계": r[1] or 0,
                "총매출금액합계": r[2] or 0,
                "반품금액합계": r[3] or 0,
                "할인금액합계": r[4] or 0,
            }
            for r in rows
        ]
    finally:
        con.close()


def run_search_mode(query: str, n_results: int, embedder: QueryEmbedder):
    where = build_where_filter(query)
    print("query =", query)
    print("where =", json.dumps(where, ensure_ascii=False) if where else None)

    query_embedding = embedder.encode_query(query)

    schema_result = search_collection(
        collection_name=SCHEMA_COLLECTION,
        query_embedding=query_embedding,
        where=None,
        n_results=min(n_results, 5),
    )
    agg_result = search_collection(
        collection_name=AGG_COLLECTION,
        query_embedding=query_embedding,
        where=where,
        n_results=n_results,
    )
    row_result = search_collection(
        collection_name=ROW_COLLECTION,
        query_embedding=query_embedding,
        where=where,
        n_results=n_results,
    )

    pretty_print_chroma_result("SCHEMA SEARCH", schema_result)
    pretty_print_chroma_result("AGG SEARCH", agg_result)
    pretty_print_chroma_result("ROW SEARCH", row_result)


def run_summary_mode(query: str, topk: int):
    print("query =", query)

    summary = duckdb_summary(query)
    by_product = duckdb_groupby_product(query, limit=topk)
    by_org = duckdb_groupby_org(query, limit=topk)
    by_date = duckdb_groupby_date(query, limit=max(topk, 31))

    print("\n===== DUCKDB SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n===== GROUP BY 제품구분 =====")
    print(json.dumps(by_product, ensure_ascii=False, indent=2))

    print("\n===== GROUP BY 조직구분 =====")
    print(json.dumps(by_org, ensure_ascii=False, indent=2))

    print("\n===== GROUP BY 일자 =====")
    print(json.dumps(by_date, ensure_ascii=False, indent=2))


def run_hybrid_mode(query: str, topk: int, embedder: QueryEmbedder):
    print("########## SEARCH ##########")
    run_search_mode(query, n_results=topk, embedder=embedder)

    print("\n\n########## SUMMARY ##########")
    run_summary_mode(query, topk=topk)


def print_help(current_mode: str, topk: int):
    print("\n사용 방법")
    print("  q) 질문입력")
    print("  q) mode search")
    print("  q) mode summary")
    print("  q) mode hybrid")
    print("  q) topk 10")
    print("  q) help")
    print("  q) exit")
    print("\n예시")
    print("  q) 20260101 과자 요약")
    print("  q) 20260101 과자 조직구분=유통 요약")
    print("  q) 관리순매출금액이 뭐야")
    print(f"\n현재 설정: mode={current_mode}, topk={topk}\n")


def interactive_loop():
    print("dbllmquery interactive mode")
    print("종료하려면 exit 입력")
    print("도움말은 help 입력\n")

    embedder = QueryEmbedder(EMBED_MODEL, use_fp16=USE_FP16, max_length=256)
    current_mode = "hybrid"
    topk = 5

    while True:
        try:
            user_input = input("q) ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not user_input:
            continue

        lowered = user_input.lower()

        if lowered in ["exit", "quit", "q"]:
            print("종료합니다.")
            break

        if lowered == "help":
            print_help(current_mode, topk)
            continue

        if lowered.startswith("mode "):
            new_mode = lowered.split(maxsplit=1)[1].strip()
            if new_mode in ["search", "summary", "hybrid"]:
                current_mode = new_mode
                print(f"mode 변경: {current_mode}")
            else:
                print("지원 모드: search / summary / hybrid")
            continue

        if lowered.startswith("topk "):
            value = lowered.split(maxsplit=1)[1].strip()
            if value.isdigit() and int(value) > 0:
                topk = int(value)
                print(f"topk 변경: {topk}")
            else:
                print("topk는 1 이상의 정수여야 합니다.")
            continue

        try:
            if current_mode == "search":
                run_search_mode(user_input, n_results=topk, embedder=embedder)
            elif current_mode == "summary":
                run_summary_mode(user_input, topk=topk)
            else:
                run_hybrid_mode(user_input, topk=topk, embedder=embedder)
        except Exception as e:
            print(f"\n[ERROR] {e}\n")


def main():
    interactive_loop()


if __name__ == "__main__":
    main()