import re
import json
import glob
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Iterable, Optional

import pandas as pd
import duckdb
import chromadb
from FlagEmbedding import BGEM3FlagModel


EMBED_MODEL = "BAAI/bge-m3"

BASE_DIR = Path("/home/siwasoft/gsllm")
INPUT_DIR = BASE_DIR / "dbxlsx"
DATA_LAKE_DIR = BASE_DIR / "data_lake"
PARQUET_DIR = DATA_LAKE_DIR / "parquet"
CHROMA_DIR = BASE_DIR / "chroma_data"
DUCKDB_PATH = BASE_DIR / "sales.duckdb"

ROW_COLLECTION = "sales_row_v1"
AGG_COLLECTION = "sales_agg_v1"
SCHEMA_COLLECTION = "sales_schema_v1"

USE_FP16 = True
ROW_EMBED_BATCH = 512
AGG_EMBED_BATCH = 1024
UPSERT_BATCH = 2000

EXPECTED_COLUMNS = [
    "제품구분",
    "조직구분",
    "일자",
    "관리조직코드",
    "거래처코드",
    "제품코드",
    "판매수량",
    "반품수량",
    "총매출금액",
    "관리순매출금액",
    "할인금액",
    "반품금액",
    "자료생성일",
]

STRING_COLUMNS = [
    "제품구분",
    "조직구분",
    "일자",
    "관리조직코드",
    "거래처코드",
    "제품코드",
    "자료생성일",
]

NUMERIC_COLUMNS = [
    "판매수량",
    "반품수량",
    "총매출금액",
    "관리순매출금액",
    "할인금액",
    "반품금액",
]


def sha256_file(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunked(seq: List, size: int) -> Iterable[List]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def normalize_date_str(x: str) -> str:
    s = str(x).strip()
    digits = re.sub(r"[^0-9]", "", s)
    if len(digits) >= 8:
        return digits[:8]
    return s


def safe_int(x) -> int:
    try:
        if pd.isna(x):
            return 0
        return int(float(x))
    except Exception:
        return 0


def safe_float(x) -> float:
    try:
        if pd.isna(x):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def format_num(x) -> str:
    v = safe_float(x)
    if abs(v - int(v)) < 1e-9:
        return str(int(v))
    return f"{v:.2f}"


def load_excel(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"헤더 누락: {missing}")

    df = df[EXPECTED_COLUMNS].copy()

    for c in STRING_COLUMNS:
        df[c] = df[c].fillna("").astype(str).str.strip()

    for c in ["일자", "자료생성일"]:
        df[c] = df[c].apply(normalize_date_str)

    for c in NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["제품구분"] = df["제품구분"].replace("", "미분류")
    df["조직구분"] = df["조직구분"].replace("", "미분류")

    return df


def init_duckdb(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    CREATE TABLE IF NOT EXISTS raw_sales (
        제품구분 VARCHAR,
        조직구분 VARCHAR,
        일자 VARCHAR,
        관리조직코드 VARCHAR,
        거래처코드 VARCHAR,
        제품코드 VARCHAR,
        판매수량 BIGINT,
        반품수량 BIGINT,
        총매출금액 DOUBLE,
        관리순매출금액 DOUBLE,
        할인금액 DOUBLE,
        반품금액 DOUBLE,
        자료생성일 VARCHAR,
        source_file VARCHAR,
        file_hash VARCHAR,
        row_hash VARCHAR,
        ingested_at TIMESTAMP
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS ingest_log (
        file_hash VARCHAR PRIMARY KEY,
        source_file VARCHAR,
        dt VARCHAR,
        row_count BIGINT,
        ingested_at TIMESTAMP
    );
    """)

    con.execute("""
    CREATE INDEX IF NOT EXISTS idx_raw_sales_dt ON raw_sales(일자);
    """)


def already_ingested(con: duckdb.DuckDBPyConnection, file_hash: str) -> bool:
    row = con.execute(
        "SELECT COUNT(*) FROM ingest_log WHERE file_hash = ?",
        [file_hash]
    ).fetchone()
    return row[0] > 0


def write_parquet(df: pd.DataFrame, dt: str, source_file: str) -> Path:
    out_dir = PARQUET_DIR / f"dt={dt}"
    out_dir.mkdir(parents=True, exist_ok=True)

    source_stem = Path(source_file).stem
    out_path = out_dir / f"{source_stem}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def make_row_hash_from_series(row: pd.Series) -> str:
    raw = "|".join([
        str(row["일자"]),
        str(row["제품구분"]),
        str(row["조직구분"]),
        str(row["관리조직코드"]),
        str(row["거래처코드"]),
        str(row["제품코드"]),
        str(safe_int(row["판매수량"])),
        str(safe_int(row["반품수량"])),
        str(safe_float(row["총매출금액"])),
        str(safe_float(row["관리순매출금액"])),
        str(safe_float(row["할인금액"])),
        str(safe_float(row["반품금액"])),
    ])
    return stable_hash(raw)


def upsert_raw_to_duckdb(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    source_file: str,
    file_hash: str,
) -> None:
    now = datetime.now().isoformat()

    tmp = df.copy()
    tmp["source_file"] = str(source_file)
    tmp["file_hash"] = file_hash
    tmp["ingested_at"] = now
    tmp["row_hash"] = [make_row_hash_from_series(row) for _, row in tmp.iterrows()]

    con.register("tmp_df", tmp)

    con.execute("""
    INSERT INTO raw_sales
    SELECT
        제품구분, 조직구분, 일자, 관리조직코드, 거래처코드, 제품코드,
        CAST(판매수량 AS BIGINT),
        CAST(반품수량 AS BIGINT),
        CAST(총매출금액 AS DOUBLE),
        CAST(관리순매출금액 AS DOUBLE),
        CAST(할인금액 AS DOUBLE),
        CAST(반품금액 AS DOUBLE),
        자료생성일,
        source_file, file_hash, row_hash, CAST(ingested_at AS TIMESTAMP)
    FROM tmp_df t
    WHERE NOT EXISTS (
        SELECT 1
        FROM raw_sales r
        WHERE r.row_hash = t.row_hash
    );
    """)

    con.unregister("tmp_df")


def insert_ingest_log(
    con: duckdb.DuckDBPyConnection,
    source_file: str,
    file_hash: str,
    dt: str,
    row_count: int,
) -> None:
    con.execute("""
    INSERT INTO ingest_log(file_hash, source_file, dt, row_count, ingested_at)
    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, [file_hash, str(source_file), dt, row_count])


def build_schema_docs() -> Tuple[List[str], List[Dict], List[str]]:
    docs = [
        "스키마 설명. 제품구분은 제품의 상위 분류이다. 예를 들어 과자, 음료 같은 분류가 들어올 수 있다.",
        "스키마 설명. 조직구분은 영업 조직 또는 채널의 분류이다. 예를 들어 유통, 직거래 등의 구분값이 들어올 수 있다.",
        "스키마 설명. 일자는 해당 영업 데이터가 발생한 기준 일자이며 YYYYMMDD 형식을 사용한다.",
        "스키마 설명. 관리조직코드, 거래처코드, 제품코드는 식별자 성격의 코드 컬럼이며 필터링과 그룹핑에 자주 사용된다.",
        "지표 설명. 판매수량은 판매된 수량이다. 반품수량은 반품된 수량이다.",
        "지표 설명. 총매출금액은 할인 및 반품 전 기준의 총매출 성격 금액이다.",
        "지표 설명. 관리순매출금액은 관리 기준의 순매출 금액이다. 업무적으로는 총매출금액, 할인금액, 반품금액과 함께 해석한다.",
        "지표 설명. 할인금액이 크면 총매출 대비 순매출 감소 영향이 있을 수 있다.",
        "지표 설명. 반품금액과 반품수량은 실적에 음수 효과를 주는 지표로 해석할 수 있다.",
        "질의응답 가이드. 제품코드, 거래처코드, 관리조직코드의 이름 매핑 테이블이 없으면 코드값 그대로 답변한다.",
    ]

    metas, ids = [], []
    for i, _ in enumerate(docs, start=1):
        metas.append({
            "doc_type": "schema",
            "topic": "schema_or_metric",
            "version": "v1",
        })
        ids.append(f"schema:{i}")
    return docs, metas, ids


def build_row_docs(df: pd.DataFrame) -> Tuple[List[str], List[Dict], List[str]]:
    docs, metas, ids = [], [], []

    for _, row in df.iterrows():
        text = (
            f"영업 원시 레코드. "
            f"일자 {row['일자']}, 제품구분 {row['제품구분']}, 조직구분 {row['조직구분']}, "
            f"관리조직코드 {row['관리조직코드']}, 거래처코드 {row['거래처코드']}, 제품코드 {row['제품코드']}, "
            f"판매수량 {safe_int(row['판매수량'])}, 반품수량 {safe_int(row['반품수량'])}, "
            f"총매출금액 {format_num(row['총매출금액'])}, 관리순매출금액 {format_num(row['관리순매출금액'])}, "
            f"할인금액 {format_num(row['할인금액'])}, 반품금액 {format_num(row['반품금액'])}."
        )

        row_hash = make_row_hash_from_series(row)
        doc_id = "row:" + row_hash

        meta = {
            "doc_type": "row",
            "일자": str(row["일자"]),
            "제품구분": str(row["제품구분"]),
            "조직구분": str(row["조직구분"]),
            "관리조직코드": str(row["관리조직코드"]),
            "거래처코드": str(row["거래처코드"]),
            "제품코드": str(row["제품코드"]),
            "자료생성일": str(row["자료생성일"]),
        }

        docs.append(text)
        metas.append(meta)
        ids.append(doc_id)

    return docs, metas, ids


def build_agg_docs_from_duckdb(
    con: duckdb.DuckDBPyConnection,
    dt: str,
) -> Tuple[List[str], List[Dict], List[str]]:
    docs, metas, ids = [], [], []

    agg_specs = [
        {"name": "by_date", "group_cols": ["일자"]},
        {"name": "by_date_product", "group_cols": ["일자", "제품구분"]},
        {"name": "by_date_org", "group_cols": ["일자", "조직구분"]},
        {"name": "by_date_product_org", "group_cols": ["일자", "제품구분", "조직구분"]},
        {"name": "by_date_product_code", "group_cols": ["일자", "제품코드"]},
        {"name": "by_date_customer_code", "group_cols": ["일자", "거래처코드"]},
    ]

    for spec in agg_specs:
        group_cols = spec["group_cols"]
        group_expr = ", ".join(group_cols)

        sql = f"""
        SELECT
            {group_expr},
            SUM(판매수량) AS 판매수량,
            SUM(반품수량) AS 반품수량,
            SUM(총매출금액) AS 총매출금액,
            SUM(관리순매출금액) AS 관리순매출금액,
            SUM(할인금액) AS 할인금액,
            SUM(반품금액) AS 반품금액,
            COUNT(*) AS row_count
        FROM raw_sales
        WHERE 일자 = ?
        GROUP BY {group_expr}
        ORDER BY {group_expr}
        """
        agg_df = con.execute(sql, [dt]).fetchdf()

        for _, row in agg_df.iterrows():
            dim_parts = []
            meta = {
                "doc_type": "agg",
                "agg_level": spec["name"],
                "일자": str(row["일자"]) if "일자" in row.index else dt,
            }

            for col in group_cols:
                value = str(row[col])
                dim_parts.append(f"{col} {value}")
                meta[col] = value

            dim_text = ", ".join(dim_parts)

            text = (
                f"영업 집계 문서. {dim_text}의 요약. "
                f"판매수량 합계 {safe_int(row['판매수량'])}, "
                f"반품수량 합계 {safe_int(row['반품수량'])}, "
                f"총매출금액 합계 {format_num(row['총매출금액'])}, "
                f"관리순매출금액 합계 {format_num(row['관리순매출금액'])}, "
                f"할인금액 합계 {format_num(row['할인금액'])}, "
                f"반품금액 합계 {format_num(row['반품금액'])}, "
                f"원시행 개수 {safe_int(row['row_count'])}."
            )

            id_raw = spec["name"] + "|" + "|".join([f"{c}={row[c]}" for c in group_cols])
            doc_id = "agg:" + stable_hash(id_raw)

            docs.append(text)
            metas.append(meta)
            ids.append(doc_id)

    return docs, metas, ids


class BGEEmbedder:
    def __init__(self, model_name: str, use_fp16: bool = True, max_length: int = 256):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.max_length = max_length

    def encode(self, texts: List[str], batch_size: int) -> List[List[float]]:
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return out["dense_vecs"]


def get_chroma_collections():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    row_col = client.get_or_create_collection(name=ROW_COLLECTION)
    agg_col = client.get_or_create_collection(name=AGG_COLLECTION)
    schema_col = client.get_or_create_collection(name=SCHEMA_COLLECTION)
    return row_col, agg_col, schema_col


def chroma_upsert(
    collection,
    ids: List[str],
    docs: List[str],
    metas: List[Dict],
    embeddings: List[List[float]],
    batch_size: int = UPSERT_BATCH,
) -> None:
    for id_batch, doc_batch, meta_batch, emb_batch in zip(
        chunked(ids, batch_size),
        chunked(docs, batch_size),
        chunked(metas, batch_size),
        chunked(embeddings, batch_size),
    ):
        collection.upsert(
            ids=id_batch,
            documents=doc_batch,
            metadatas=meta_batch,
            embeddings=emb_batch,
        )


def ingest_one_file(
    file_path: str,
    con: duckdb.DuckDBPyConnection,
    embedder_row: BGEEmbedder,
    embedder_agg: BGEEmbedder,
    force: bool = False,
) -> Dict:
    file_hash = sha256_file(file_path)

    if not force and already_ingested(con, file_hash):
        return {
            "file": file_path,
            "status": "skipped",
            "reason": "same file hash already ingested",
        }

    df = load_excel(file_path)

    if df.empty:
        return {
            "file": file_path,
            "status": "skipped",
            "reason": "empty dataframe",
        }

    unique_dates = sorted(df["일자"].dropna().unique().tolist())
    if len(unique_dates) != 1:
        raise ValueError(f"한 파일에 일자가 여러 개 존재합니다: {unique_dates}")

    dt = str(unique_dates[0])

    parquet_path = write_parquet(df, dt, file_path)
    upsert_raw_to_duckdb(con, df, file_path, file_hash)

    row_col, agg_col, schema_col = get_chroma_collections()

    schema_docs, schema_metas, schema_ids = build_schema_docs()
    schema_embs = []
    for batch in chunked(schema_docs, AGG_EMBED_BATCH):
        schema_embs.extend(embedder_agg.encode(batch, batch_size=len(batch)))
    chroma_upsert(schema_col, schema_ids, schema_docs, schema_metas, schema_embs)

    row_docs, row_metas, row_ids = build_row_docs(df)
    row_embs = []
    for batch in chunked(row_docs, ROW_EMBED_BATCH):
        row_embs.extend(embedder_row.encode(batch, batch_size=len(batch)))
    chroma_upsert(row_col, row_ids, row_docs, row_metas, row_embs)

    agg_docs, agg_metas, agg_ids = build_agg_docs_from_duckdb(con, dt)
    agg_embs = []
    for batch in chunked(agg_docs, AGG_EMBED_BATCH):
        agg_embs.extend(embedder_agg.encode(batch, batch_size=len(batch)))
    chroma_upsert(agg_col, agg_ids, agg_docs, agg_metas, agg_embs)

    insert_ingest_log(con, file_path, file_hash, dt, len(df))

    return {
        "file": file_path,
        "status": "ingested",
        "dt": dt,
        "rows": len(df),
        "parquet": str(parquet_path),
        "row_docs": len(row_docs),
        "agg_docs": len(agg_docs),
        "schema_docs": len(schema_docs),
    }


def resolve_input_files(input_path: str) -> List[str]:
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        return sorted(glob.glob(str(p / "*.xlsx")))
    raise FileNotFoundError(f"입력 경로를 찾을 수 없습니다: {input_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(INPUT_DIR),
        help="xlsx 파일 또는 xlsx들이 들어있는 디렉토리",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="이미 ingest된 파일도 다시 처리",
    )
    args = parser.parse_args()

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    files = resolve_input_files(args.input)
    if not files:
        print("처리할 xlsx 파일이 없습니다.")
        return

    con = duckdb.connect(str(DUCKDB_PATH))
    init_duckdb(con)

    embedder_row = BGEEmbedder(EMBED_MODEL, use_fp16=USE_FP16, max_length=256)
    embedder_agg = BGEEmbedder(EMBED_MODEL, use_fp16=USE_FP16, max_length=256)

    results = []
    for file_path in files:
        try:
            result = ingest_one_file(
                file_path=file_path,
                con=con,
                embedder_row=embedder_row,
                embedder_agg=embedder_agg,
                force=args.force,
            )
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))
        except Exception as e:
            error = {
                "file": file_path,
                "status": "error",
                "error": str(e),
            }
            results.append(error)
            print(json.dumps(error, ensure_ascii=False))

    print("\n=== summary ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()