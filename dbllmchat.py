import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import chromadb
from FlagEmbedding import BGEM3FlagModel
from llama_cpp import Llama


# =========================
# Paths / Collections
# =========================
BASE_DIR = Path("/home/siwasoft/gsllm")
CHROMA_DIR = BASE_DIR / "chroma_data"
DUCKDB_PATH = BASE_DIR / "sales.duckdb"

ROW_COLLECTION = "sales_row_v1"
AGG_COLLECTION = "sales_agg_v1"
SCHEMA_COLLECTION = "sales_schema_v1"

EMBED_MODEL = "BAAI/bge-m3"
USE_FP16 = True

LLM_MODEL = "/home/siwasoft/gsllm/gemma3-27b/gemma-3-27b-it-Q4_K_M.gguf"

KNOWN_PRODUCT_TYPES = ["과자", "음료", "빙과", "식품", "기타", "미분류"]
KNOWN_ORG_TYPES = ["유통", "직거래", "특판", "온라인", "대리점", "수출", "미분류"]

BUSINESS_KEYWORDS = [
    "매출", "순매출", "판매", "판매수량", "반품", "반품금액", "반품수량",
    "할인", "할인금액", "제품", "제품구분", "제품코드",
    "거래처", "거래처코드", "조직", "조직구분", "관리조직코드",
    "일자", "데이터", "영업", "실적", "요약", "합계", "비교",
    "사례", "예시", "거래", "top", "상위", "하위", "모든행"
]


# =========================
# Embedding / LLM
# =========================
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


class LocalLLM:
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=8192,
            n_gpu_layers=-1,
            n_threads=8,
            verbose=False,
        )

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        out = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 영업 데이터 분석 챗봇이다. "
                        "제공된 숫자와 근거만 사용하고 추측하지 마라. "
                        "질문 의도를 벗어나지 말고, 불필요한 일반 대화를 하지 마라."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.0,
            max_tokens=max_new_tokens,
        )
        return out["choices"][0]["message"]["content"].strip()


# =========================
# Query parsing
# =========================
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


def extract_top_n(text: str, default: int = 5) -> int:
    m = re.search(r"(top\s*\d+|상위\s*\d+|하위\s*\d+|\d+\s*개)", text.lower())
    if m:
        digits = re.findall(r"\d+", m.group(0))
        if digits:
            n = int(digits[0])
            return max(1, min(n, 100))
    return default


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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def is_business_query(query: str) -> bool:
    q = normalize_text(query)
    if extract_dates(query):
        return True
    if extract_known_value(query, KNOWN_PRODUCT_TYPES):
        return True
    if extract_explicit_org_type(query):
        return True
    return any(k in q for k in BUSINESS_KEYWORDS)


def detect_query_type(query: str) -> str:
    q = normalize_text(query)

    if not is_business_query(query):
        return "reject"

    if "모든행" in q or "전체 행" in q or "전부 보여" in q:
        return "all_rows"

    if any(x in q for x in ["상위", "top", "큰 거래", "큰 사례", "가장 큰", "높은 거래"]):
        return "top_rows"

    if "반품" in q and any(x in q for x in ["사례", "예시", "거래", "보여줘", "보여 줘"]):
        return "return_examples"

    if any(x in q for x in ["순매출이 약", "순매출이 낮", "남는 게 적", "순매출 약한 거래"]):
        return "weak_net_examples"

    if "할인" in q and any(x in q for x in ["사례", "예시", "거래", "보여줘", "보여 줘"]):
        return "discount_examples"

    if any(x in q for x in ["사례", "예시", "거래 보여", "샘플", "레코드", "row"]):
        return "row_examples"

    if any(x in q for x in ["요약", "합계", "총합", "비교", "얼마", "몇", "비중"]):
        return "summary"

    if any(x in q for x in ["뭐야", "의미", "설명", "정의"]):
        return "schema"

    return "summary"


# =========================
# Chroma
# =========================
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
    return col.query(
        query_embeddings=[query_embedding],
        where=where,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )


def flatten_results(result: Dict) -> List[Dict]:
    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0] if "distances" in result else []

    rows = []
    for i in range(len(ids)):
        rows.append({
            "id": ids[i],
            "document": docs[i] if i < len(docs) else "",
            "metadata": metas[i] if i < len(metas) else {},
            "distance": dists[i] if i < len(dists) else None,
        })
    return rows


# =========================
# DuckDB
# =========================
def connect_duckdb():
    return duckdb.connect(str(DUCKDB_PATH), read_only=True)


def duckdb_summary(query: str) -> Dict:
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


def duckdb_groupby_product(query: str, limit: int = 10) -> List[Dict]:
    con = connect_duckdb()
    try:
        where_sql, params = build_sql_filters_from_query(query)
        sql = f"""
        SELECT
            제품구분,
            SUM(판매수량) AS 판매수량합계,
            SUM(총매출금액) AS 총매출금액합계,
            SUM(관리순매출금액) AS 관리순매출금액합계,
            SUM(할인금액) AS 할인금액합계,
            SUM(반품금액) AS 반품금액합계
        FROM raw_sales
        {where_sql}
        GROUP BY 1
        ORDER BY 관리순매출금액합계 DESC
        LIMIT {limit}
        """
        rows = con.execute(sql, params).fetchall()
        return [
            {
                "제품구분": r[0],
                "판매수량합계": r[1] or 0,
                "총매출금액합계": r[2] or 0,
                "관리순매출금액합계": r[3] or 0,
                "할인금액합계": r[4] or 0,
                "반품금액합계": r[5] or 0,
            }
            for r in rows
        ]
    finally:
        con.close()


def duckdb_groupby_org(query: str, limit: int = 10) -> List[Dict]:
    con = connect_duckdb()
    try:
        where_sql, params = build_sql_filters_from_query(query)
        sql = f"""
        SELECT
            조직구분,
            SUM(판매수량) AS 판매수량합계,
            SUM(총매출금액) AS 총매출금액합계,
            SUM(관리순매출금액) AS 관리순매출금액합계,
            SUM(할인금액) AS 할인금액합계,
            SUM(반품금액) AS 반품금액합계
        FROM raw_sales
        {where_sql}
        GROUP BY 1
        ORDER BY 관리순매출금액합계 DESC
        LIMIT {limit}
        """
        rows = con.execute(sql, params).fetchall()
        return [
            {
                "조직구분": r[0],
                "판매수량합계": r[1] or 0,
                "총매출금액합계": r[2] or 0,
                "관리순매출금액합계": r[3] or 0,
                "할인금액합계": r[4] or 0,
                "반품금액합계": r[5] or 0,
            }
            for r in rows
        ]
    finally:
        con.close()


def duckdb_groupby_date(query: str, limit: int = 31) -> List[Dict]:
    con = connect_duckdb()
    try:
        where_sql, params = build_sql_filters_from_query(query)
        sql = f"""
        SELECT
            일자,
            SUM(판매수량) AS 판매수량합계,
            SUM(총매출금액) AS 총매출금액합계,
            SUM(관리순매출금액) AS 관리순매출금액합계,
            SUM(할인금액) AS 할인금액합계,
            SUM(반품금액) AS 반품금액합계
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
                "판매수량합계": r[1] or 0,
                "총매출금액합계": r[2] or 0,
                "관리순매출금액합계": r[3] or 0,
                "할인금액합계": r[4] or 0,
                "반품금액합계": r[5] or 0,
            }
            for r in rows
        ]
    finally:
        con.close()


def duckdb_top_discount_rows(query: str, limit: int = 5) -> List[Dict]:
    con = connect_duckdb()
    try:
        where_sql, params = build_sql_filters_from_query(query)
        extra = "AND 할인금액 > 0" if where_sql else "WHERE 할인금액 > 0"
        sql = f"""
        SELECT
            일자, 제품구분, 조직구분, 관리조직코드, 거래처코드, 제품코드,
            판매수량, 반품수량, 총매출금액, 관리순매출금액, 할인금액, 반품금액
        FROM raw_sales
        {where_sql}
        {extra}
        ORDER BY 할인금액 DESC, 관리순매출금액 DESC
        LIMIT {limit}
        """
        rows = con.execute(sql, params).fetchall()
        return rows_to_dicts(rows)
    finally:
        con.close()


def duckdb_return_rows(query: str, limit: int = 5) -> List[Dict]:
    con = connect_duckdb()
    try:
        where_sql, params = build_sql_filters_from_query(query)
        extra = "AND (반품수량 > 0 OR 반품금액 > 0)" if where_sql else "WHERE (반품수량 > 0 OR 반품금액 > 0)"
        sql = f"""
        SELECT
            일자, 제품구분, 조직구분, 관리조직코드, 거래처코드, 제품코드,
            판매수량, 반품수량, 총매출금액, 관리순매출금액, 할인금액, 반품금액
        FROM raw_sales
        {where_sql}
        {extra}
        ORDER BY 반품금액 DESC, 반품수량 DESC, 총매출금액 DESC
        LIMIT {limit}
        """
        rows = con.execute(sql, params).fetchall()
        return rows_to_dicts(rows)
    finally:
        con.close()


def duckdb_weak_net_rows(query: str, limit: int = 5) -> List[Dict]:
    con = connect_duckdb()
    try:
        where_sql, params = build_sql_filters_from_query(query)
        extra = "AND 총매출금액 > 0" if where_sql else "WHERE 총매출금액 > 0"
        sql = f"""
        SELECT
            일자, 제품구분, 조직구분, 관리조직코드, 거래처코드, 제품코드,
            판매수량, 반품수량, 총매출금액, 관리순매출금액, 할인금액, 반품금액,
            CASE WHEN 총매출금액 > 0 THEN 관리순매출금액 / 총매출금액 ELSE NULL END AS 순매출비율
        FROM raw_sales
        {where_sql}
        {extra}
        ORDER BY 순매출비율 ASC NULLS LAST, 총매출금액 DESC
        LIMIT {limit}
        """
        rows = con.execute(sql, params).fetchall()
        return rows_to_dicts(rows, include_ratio=True)
    finally:
        con.close()


def duckdb_all_rows(query: str, limit: int = 20) -> List[Dict]:
    con = connect_duckdb()
    try:
        where_sql, params = build_sql_filters_from_query(query)
        sql = f"""
        SELECT
            일자, 제품구분, 조직구분, 관리조직코드, 거래처코드, 제품코드,
            판매수량, 반품수량, 총매출금액, 관리순매출금액, 할인금액, 반품금액
        FROM raw_sales
        {where_sql}
        ORDER BY 일자, 제품구분, 조직구분, 거래처코드, 제품코드
        LIMIT {limit}
        """
        rows = con.execute(sql, params).fetchall()
        return rows_to_dicts(rows)
    finally:
        con.close()


def rows_to_dicts(rows, include_ratio: bool = False) -> List[Dict]:
    out = []
    for r in rows:
        item = {
            "일자": r[0],
            "제품구분": r[1],
            "조직구분": r[2],
            "관리조직코드": r[3],
            "거래처코드": r[4],
            "제품코드": r[5],
            "판매수량": r[6],
            "반품수량": r[7],
            "총매출금액": r[8],
            "관리순매출금액": r[9],
            "할인금액": r[10],
            "반품금액": r[11],
        }
        if include_ratio and len(r) > 12:
            item["순매출비율"] = r[12]
        out.append(item)
    return out


# =========================
# Context builders
# =========================
def build_summary_context(query: str) -> Dict:
    return {
        "query_type": "summary",
        "duckdb_summary": duckdb_summary(query),
        "groupby_product": duckdb_groupby_product(query, limit=10),
        "groupby_org": duckdb_groupby_org(query, limit=10),
        "groupby_date": duckdb_groupby_date(query, limit=31),
    }


def build_schema_context(query: str, embedder: QueryEmbedder, limit: int = 5) -> Dict:
    where = None
    query_embedding = embedder.encode_query(query)
    schema_hits = flatten_results(
        search_collection(SCHEMA_COLLECTION, query_embedding, where=where, n_results=limit)
    )
    return {
        "query_type": "schema",
        "schema_hits": schema_hits,
    }


def build_row_context(query: str, embedder: QueryEmbedder, limit: int = 5) -> Dict:
    where = build_where_filter(query)
    query_embedding = embedder.encode_query(query)
    row_hits = flatten_results(
        search_collection(ROW_COLLECTION, query_embedding, where=where, n_results=limit)
    )
    agg_hits = flatten_results(
        search_collection(AGG_COLLECTION, query_embedding, where=where, n_results=min(limit, 5))
    )
    return {
        "query_type": "row_examples",
        "where": where,
        "row_hits": row_hits,
        "agg_hits": agg_hits,
    }


def build_discount_examples_context(query: str, limit: int) -> Dict:
    return {
        "query_type": "discount_examples",
        "rows": duckdb_top_discount_rows(query, limit=limit),
    }


def build_return_examples_context(query: str, limit: int) -> Dict:
    return {
        "query_type": "return_examples",
        "rows": duckdb_return_rows(query, limit=limit),
    }


def build_weak_net_context(query: str, limit: int) -> Dict:
    return {
        "query_type": "weak_net_examples",
        "rows": duckdb_weak_net_rows(query, limit=limit),
    }


def build_all_rows_context(query: str, limit: int) -> Dict:
    return {
        "query_type": "all_rows",
        "limit_applied": limit,
        "rows": duckdb_all_rows(query, limit=limit),
    }


def build_top_rows_context(query: str, limit: int) -> Dict:
    q = normalize_text(query)

    if "할인" in q:
        rows = duckdb_top_discount_rows(query, limit=limit)
        metric = "할인금액"
    elif "반품" in q:
        rows = duckdb_return_rows(query, limit=limit)
        metric = "반품금액/반품수량"
    elif "순매출" in q or "남는 게 적" in q:
        rows = duckdb_weak_net_rows(query, limit=limit)
        metric = "순매출비율이 낮은 순"
    else:
        rows = duckdb_top_discount_rows(query, limit=limit)
        metric = "할인금액"

    return {
        "query_type": "top_rows",
        "metric": metric,
        "rows": rows,
    }


# =========================
# Prompt builders
# =========================
def build_prompt_for_summary(query: str, context: Dict) -> str:
    return f"""
[질문]
{query}

[규칙]
- duckdb_summary와 groupby 결과만 사용해서 답하라.
- 숫자는 있는 그대로 써라.
- 질문 조건을 먼저 짧게 설명하라.
- 마지막에 근거를 한 줄로 적어라.

[context]
{json.dumps(context, ensure_ascii=False, indent=2)}

[답변]
""".strip()


def build_prompt_for_schema(query: str, context: Dict) -> str:
    return f"""
[질문]
{query}

[규칙]
- schema_hits만 이용해서 설명하라.
- 숫자 추정 금지.
- 짧고 명확하게 답하라.
- 마지막에 근거를 한 줄로 적어라.

[context]
{json.dumps(context, ensure_ascii=False, indent=2)}

[답변]
""".strip()


def build_prompt_for_rows(query: str, context: Dict) -> str:
    return f"""
[질문]
{query}

[규칙]
- row_hits 안의 실제 사례를 중심으로 답하라.
- 사용자가 예시/사례를 원하면 3~5개 정도를 bullet 없이 자연스럽게 정리하라.
- 없는 정보는 추정하지 마라.
- 마지막에 근거를 한 줄로 적어라.

[context]
{json.dumps(context, ensure_ascii=False, indent=2)}

[답변]
""".strip()


def build_prompt_for_sql_rows(query: str, context: Dict) -> str:
    return f"""
[질문]
{query}

[규칙]
- rows 안의 결과만 사용하라.
- 사용자가 사례/상위/모든행을 물었으면 rows를 그대로 충실하게 요약하라.
- '모든행'이라도 limit_applied가 있으면 제한이 걸렸다고 분명히 말하라.
- 마지막에 근거를 한 줄로 적어라.

[context]
{json.dumps(context, ensure_ascii=False, indent=2)}

[답변]
""".strip()


# =========================
# Answer routing
# =========================
def answer_question(query: str, embedder: QueryEmbedder, llm: LocalLLM, row_limit: int = 20) -> Tuple[str, Dict]:
    qtype = detect_query_type(query)
    top_n = extract_top_n(query, default=5)

    if qtype == "reject":
        return "현재 영업데이터 질의만 지원합니다. 일자, 제품구분, 조직구분, 매출, 순매출, 할인, 반품, 거래 사례 같은 질문으로 입력해 주세요.", {
            "query_type": "reject"
        }

    if qtype == "summary":
        context = build_summary_context(query)
        prompt = build_prompt_for_summary(query, context)
        return llm.generate(prompt, max_new_tokens=500), context

    if qtype == "schema":
        context = build_schema_context(query, embedder=embedder, limit=5)
        prompt = build_prompt_for_schema(query, context)
        return llm.generate(prompt, max_new_tokens=300), context

    if qtype == "row_examples":
        context = build_row_context(query, embedder=embedder, limit=min(top_n, 10))
        prompt = build_prompt_for_rows(query, context)
        return llm.generate(prompt, max_new_tokens=500), context

    if qtype == "discount_examples":
        context = build_discount_examples_context(query, limit=top_n)
        prompt = build_prompt_for_sql_rows(query, context)
        return llm.generate(prompt, max_new_tokens=500), context

    if qtype == "return_examples":
        context = build_return_examples_context(query, limit=top_n)
        prompt = build_prompt_for_sql_rows(query, context)
        return llm.generate(prompt, max_new_tokens=500), context

    if qtype == "weak_net_examples":
        context = build_weak_net_context(query, limit=top_n)
        prompt = build_prompt_for_sql_rows(query, context)
        return llm.generate(prompt, max_new_tokens=500), context

    if qtype == "top_rows":
        context = build_top_rows_context(query, limit=top_n)
        prompt = build_prompt_for_sql_rows(query, context)
        return llm.generate(prompt, max_new_tokens=500), context

    if qtype == "all_rows":
        context = build_all_rows_context(query, limit=row_limit)
        prompt = build_prompt_for_sql_rows(query, context)
        return llm.generate(prompt, max_new_tokens=700), context

    context = build_summary_context(query)
    prompt = build_prompt_for_summary(query, context)
    return llm.generate(prompt, max_new_tokens=500), context


# =========================
# CLI
# =========================
def print_help(row_limit: int, debug: bool):
    print("\n사용 방법")
    print("  q) 질문")
    print("  q) limit 20")
    print("  q) debug on")
    print("  q) debug off")
    print("  q) help")
    print("  q) exit")
    print("\n예시")
    print("  q) 20260101 과자 요약")
    print("  q) 20260101 과자 조직구분=유통 요약")
    print("  q) 관리순매출금액이 뭐야")
    print("  q) 할인금액이 큰 거래 5개 보여줘")
    print("  q) 반품 거래 예시 보여줘")
    print("  q) 순매출이 약한 거래 보여줘")
    print("  q) 20260101 과자 모든행 보여줘")
    print(f"\n현재 설정: limit={row_limit}, debug={debug}\n")


def interactive_loop():
    print("dbllmchat interactive mode")
    print("종료하려면 exit 입력")
    print("도움말은 help 입력\n")

    embedder = QueryEmbedder(EMBED_MODEL, use_fp16=USE_FP16, max_length=256)
    llm = LocalLLM(LLM_MODEL)

    row_limit = 20
    debug = False

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
            print_help(row_limit, debug)
            continue

        if lowered == "debug on":
            debug = True
            print("debug = on")
            continue

        if lowered == "debug off":
            debug = False
            print("debug = off")
            continue

        if lowered.startswith("limit "):
            value = lowered.split(maxsplit=1)[1].strip()
            if value.isdigit() and int(value) > 0:
                row_limit = min(int(value), 200)
                print(f"limit 변경: {row_limit}")
            else:
                print("limit는 1 이상의 정수여야 합니다.")
            continue

        try:
            answer, context = answer_question(
                user_input,
                embedder=embedder,
                llm=llm,
                row_limit=row_limit,
            )

            if debug:
                print("\n[DEBUG CONTEXT]")
                print(json.dumps(context, ensure_ascii=False, indent=2))

            print("\n=== ANSWER ===")
            print(answer)
            print()
        except Exception as e:
            print(f"\n[ERROR] {e}\n")


def main():
    interactive_loop()


if __name__ == "__main__":
    main()