#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
poc_snacks_cli_router_redesigned.py

재설계 버전
- 질문 해석 / intent 결정 / 실행 / 출력 분리
- 기간 판정 개선 (단일 YYYY-MM-DD 를 기간으로 오인하지 않음)
- 룰 기반 intent table 적용
- DateResolution 구조화
- SAFE PandasPlan fallback 유지
"""

import os
import re
import gc
import json
import argparse
import calendar
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Optional, Tuple, Dict, List, Literal, Callable

import pandas as pd

try:
    import chromadb  # type: ignore[import-not-found]
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction  # type: ignore[import-not-found]
except Exception:
    chromadb = None
    DefaultEmbeddingFunction = None

try:
    from rapidfuzz import process as rf_process  # type: ignore[import-not-found]
    from rapidfuzz import fuzz as rf_fuzz  # type: ignore[import-not-found]
except Exception:
    rf_process = None
    rf_fuzz = None


# ============================================================
# Defaults / Config
# ============================================================
DEFAULT_CHROMA_PATH = "./chroma_snacks"
DEFAULT_COLL_SUMMARY = "snack_price_compare_v1"
DEFAULT_COLL_REVIEWS = "snack_reviews_digest_v1"

DEFAULT_OUTPUT_DIR = "./outputs"

MY_MANUFACTURER = "해태제과"
MY_WORDS = [
    "자사", "우리", "당사",
    "우리회사", "우리 회사", "본사",
    "우리제품", "우리 제품", "당사제품", "당사 제품",
]

DEFAULT_ENABLE_LLM = True
DEFAULT_PRINT_PLAN = True
DEFAULT_LLM_MODEL_PATH = "/home/siwasoft/gsllm/exaone-4.0-32b/EXAONE-4.0-32B-Instruct-Q4_K_M.gguf"
DEFAULT_LLM_N_CTX = 16384
DEFAULT_LLM_MAX_TOKENS = 900
DEFAULT_LLM_TEMPERATURE = 0.12
DEFAULT_LLM_THREADS = 16
DEFAULT_LLM_GPU_LAYERS = 20

MAX_RESULT_ROWS_PRINT = 2000
MAX_RESULT_COLS_PRINT = 200
CSV_SAVE_THRESHOLD = 600
HARD_LIMIT_ROWS = 20000

DEFAULT_RAW_LIMIT = 300
DEFAULT_SUMMARY_LIMIT = 50

FUZZY_CUTOFF = 88

SHIP_WORDS = ["배송", "지연", "늦", "파손", "오배송", "택배", "배송비"]
SHIP_NEG_WORDS = ["불만", "문제", "이슈", "클레임", "컴플", "항의", "불편", "민원"]

COMPARE_WORDS = [
    "비교", "대비", "vs", "VS", "versus", "차이", "갭", "gap",
    "더 비싼", "더 싼", "누가 더", "어디가 더", "비교해서",
    "비교하면", "비교해줘", "비교해", "쪽이 더", "어느 쪽", "어느쪽"
]

DETAIL_RAW_DEFAULT_SELECT = [
    "batch_date", "manufacturer", "mall_name", "product_key", "product_name",
    "rank", "price", "item_name", "url", "comments_top5", "date"
]

SUMMARY_DEFAULT_SELECT = [
    "batch_date", "manufacturer", "mall_name", "product_key", "product_name",
    "min_price", "avg_price", "max_price"
]

PRODUCT_QUERY_HINTS = [
    "가격", "최저가", "최고가", "평균가", "시세",
    "싼", "저렴", "비싼",
    "오퍼", "상세", "추이", "원본",
    "제일 싼", "가장 싼", "가장 저렴", "최저", "최고", "평균"
]

COLNAME_KR: Dict[str, str] = {
    "batch_date": "배치일자",
    "manufacturer": "제조사",
    "mall_name": "쇼핑몰",
    "target_mall": "기준몰",
    "cheaper_mall": "더저렴한몰",
    "product_key": "제품코드",
    "product_name": "제품명",
    "min_price": "최저가",
    "avg_price": "평균가",
    "max_price": "최고가",
    "rank": "순위",
    "price": "가격",
    "url": "URL",
    "item_name": "상품명",
    "comments_top5": "상위댓글5",
    "date": "수집시각",
    "ship_issue_mentions": "배송이슈언급수",
    "rows_count": "오퍼건수",
    "rate": "이슈비율",
    "cheapest_mall": "최저가몰",
    "cheapest_price": "최저가금액",
    "cheaper_price": "더저렴한가격",
    "target_mall_price": "타겟몰가격",
    "diff": "차액",
    "violations_count": "위반건수",
    "sum_diff": "차액합계",
    "avg_diff": "평균차액",
    "total_products": "전체제품수",
    "global_min_price": "전체최저가",
    "value": "값",
    "min_price_min": "기간최저가(최소)",
    "max_price_max": "기간최고가(최대)",
    "product_count": "제품수",
    "case_count": "케이스수",
    "mall_count": "몰수",
    "manufacturer_count": "제조사수",
}

VIOLATION_WORDS = [
    "위반", "정책 위반", "규정 위반", "컴플", "컴플라이언스", "페널티",
    "보상", "환급", "배상", "물어내", "물어 내"
]
DETAIL_WORDS_EXT = [
    "상세", "내역", "목록", "리스트", "전부", "전체", "원본",
    "항목", "건별", "케이스", "사례", "품목", "상품", "제품", "라인업",
    "위반건", "위반 건", "세부", "디테일"
]
TREND_WORDS_EXT = [
    "추이", "트렌드", "변화", "변동", "흐름", "추세", "그래프", "차트",
    "일별", "날짜별", "기간별", "타임라인", "timeline"
]
OFFER_WORDS = [
    "오퍼", "목록", "리스트", "전체", "전부", "원본", "raw",
    "offers", "전건", "원문", "원자료", "랭크", "rank", "1등", "1위", "top1"
]
PRICE_WORDS_EXT = [
    "가격", "시세", "얼마", "몇원", "몇 원", "최저가", "최고가", "평균가",
    "최저", "평균", "최대", "저렴", "싼", "비싼", "가격 정보", "가격 알려", "가격 좀",
    "싼 것들", "저렴한 것들", "비싼 것들", "높은 것들", "낮은 것들"
]
TOP_WORDS_EXT = ["상위", "하위", "top", "TOP", "탑", "베스트", "랭킹", "순위"]

DASH_VARIANTS = ["–", "—", "−"]
WAVE_VARIANTS = ["∼", "〜"]

PLAN_FORBIDDEN_PATTERNS = [
    "위반", "보상", "배상", "정책 위반", "규정 위반", "컴플",
    "최저가 아닌", "최저가 아님", "더 싼", "더 저렴한",
    "최저가 놓친", "타겟몰보다 낮은", "다른 곳이 더 싸",
    "비교 위반", "물어내", "환급"
]

LLM = None


# ============================================================
# Data classes
# ============================================================
@dataclass
class DateResolution:
    requested_start: str
    requested_end: str
    resolved_start: str
    resolved_end: str
    is_range: bool
    clamped_start: bool
    clamped_end: bool
    source: str  # single_date | explicit_range | relative_range | calendar_range

    @property
    def batch_date(self) -> str:
        return self.resolved_end if self.is_range else self.resolved_start


@dataclass
class ParsedEntities:
    product_key: Optional[str]
    mall: Optional[str]
    manufacturer: Optional[str]
    mall_list: List[str]
    manufacturer_list: List[str]


@dataclass
class SemanticSlots:
    has_violation: bool
    wants_detail: bool
    wants_trend: bool
    wants_offer: bool
    wants_price: bool
    wants_topn: bool
    wants_shipping_issue: bool
    has_metric: bool
    has_range_expr: bool
    not_cheapest: bool
    cheapest_only: bool
    wants_date_in_rows: bool
    prefers_summary: bool
    prefers_raw: bool
    wants_all_rows: bool
    wants_group_by_mall: bool
    has_generic_price_bucket_words: bool
    has_compare_words: bool
    view_source: str


@dataclass
class QueryContext:
    raw_question: str
    question: str
    entities: ParsedEntities
    dates: DateResolution
    slots: SemanticSlots
    intent: str


@dataclass
class AppContext:
    args: argparse.Namespace
    df_data_all: pd.DataFrame
    df_pc_all: pd.DataFrame
    catalog: Dict[str, Any]
    today: str
    default_start: str


# ============================================================
# SAFE PandasPlan
# ============================================================
PandasSource = Literal["df_pc_all", "df_data_all"]
FilterOp = Literal["=", "!=", "contains", "between", "in", "not in"]
AggFunc = Literal["count", "sum", "mean", "min", "max", "nunique"]
SortDir = Literal["asc", "desc"]


@dataclass
class PandasFilter:
    col: str
    op: FilterOp
    value: Optional[Any] = None
    start: Optional[str] = None
    end: Optional[str] = None


@dataclass
class PandasAgg:
    col: str
    func: AggFunc
    as_name: str


@dataclass
class PandasSort:
    col: str
    dir: SortDir


@dataclass
class PandasPlan:
    source: PandasSource
    select: List[str]
    filters: List[PandasFilter]
    groupby: List[str]
    aggregations: List[PandasAgg]
    sort: List[PandasSort]
    limit: int


# ============================================================
# Utils
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def now_ts() -> int:
    return int(datetime.now().timestamp())


def normalize_range_separators(q: str) -> str:
    t = q
    for d in DASH_VARIANTS:
        t = t.replace(d, "-")
    for w in WAVE_VARIANTS:
        t = t.replace(w, "~")
    t = re.sub(r"\bto\b", "-", t, flags=re.IGNORECASE)
    return t


def normalize_lookup_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip().upper()
    t = re.sub(r"[\s\-_./]+", "", t)
    t = re.sub(r"[^A-Z0-9가-힣]", "", t)
    return t


def normalize_batch_date_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip().str.slice(0, 10)
    out = out.str.replace("/", "-", regex=False)
    return out


def today_from_df(df_pc: pd.DataFrame) -> str:
    if "batch_date" not in df_pc.columns or len(df_pc) == 0:
        return ""
    return str(df_pc["batch_date"].astype(str).max())[:10]


def min_date_from_df(df_pc: pd.DataFrame, fallback: str) -> str:
    try:
        m = str(df_pc["batch_date"].astype(str).min())[:10]
        return m if m else fallback
    except Exception:
        return fallback


def shift_date_ymd(date_str: str, delta_days: int) -> str:
    try:
        dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
        return (dt + timedelta(days=int(delta_days))).strftime("%Y-%m-%d")
    except Exception:
        return str(date_str)[:10]


def expand_date_range_days(start_date: str, end_date: str, max_days: int = 45) -> List[str]:
    try:
        s = datetime.strptime(str(start_date)[:10], "%Y-%m-%d")
        e = datetime.strptime(str(end_date)[:10], "%Y-%m-%d")
    except Exception:
        if start_date != end_date:
            return [str(start_date)[:10], str(end_date)[:10]]
        return [str(start_date)[:10]]

    if s > e:
        s, e = e, s

    days = (e - s).days + 1
    if days <= 0:
        return [s.strftime("%Y-%m-%d")]
    if days > max_days:
        return [s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")]

    return [(s + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]


def _month_range_for(dt: datetime) -> Tuple[str, str]:
    y, m = dt.year, dt.month
    last = calendar.monthrange(y, m)[1]
    return f"{y:04d}-{m:02d}-01", f"{y:04d}-{m:02d}-{last:02d}"


def _week_range_for(dt: datetime) -> Tuple[str, str]:
    start = dt - timedelta(days=dt.weekday())
    end = start + timedelta(days=6)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def ship_issue_count(text: Any) -> int:
    if not isinstance(text, str):
        return 0
    return sum(text.count(w) for w in SHIP_WORDS)


def has_my_words(q: str) -> bool:
    return any(w in q for w in MY_WORDS)


def df_for_terminal(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return df.rename(columns=COLNAME_KR).copy()
    except Exception:
        return df.copy()


def df_to_string_kr(df: pd.DataFrame, index: bool = False) -> str:
    return df_for_terminal(df).to_string(index=index)


def print_debug_json(payload: Dict[str, Any]):
    try:
        print("[DEBUG_JSON]", json.dumps(payload, ensure_ascii=False))
    except Exception:
        print("[DEBUG_JSON]", str(payload))


def print_result_any(result: Any, output_dir: str, prefix: str = "RESULT"):
    if isinstance(result, pd.DataFrame):
        df = result.copy()
        too_many_cols = df.shape[1] > MAX_RESULT_COLS_PRINT
        too_many_rows = df.shape[0] > MAX_RESULT_ROWS_PRINT

        if df.shape[0] >= CSV_SAVE_THRESHOLD or too_many_rows or too_many_cols:
            ensure_dir(output_dir)
            ts = now_ts()
            path = os.path.join(output_dir, f"{prefix}_{ts}.csv")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"[결과] DataFrame이 커서 CSV로 저장했습니다: {path}")
            print(df_to_string_kr(df.head(50), index=False))
            print(f"... (총 {len(df)}행, {df.shape[1]}열)")
            return

        print(df_to_string_kr(df, index=False))
        return

    try:
        if isinstance(result, (dict, list)):
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(result)
    except Exception:
        print(result)


# ============================================================
# Fuzzy
# ============================================================
def _fuzzy_pick(query: str, choices: List[str], cutoff: int = FUZZY_CUTOFF) -> Optional[str]:
    if not query or not choices:
        return None
    if rf_process is None or rf_fuzz is None:
        return None
    try:
        hit = rf_process.extractOne(query, choices, scorer=rf_fuzz.WRatio, score_cutoff=cutoff)
        if not hit:
            return None
        return str(hit[0])
    except Exception:
        return None


# ============================================================
# Excel loader
# ============================================================
def _extract_date_from_filename(path: str) -> Optional[str]:
    base = os.path.basename(path)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", base)
    return m.group(1) if m else None


def load_excel_one(xlsx_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ds = _extract_date_from_filename(xlsx_path)

    sheet_data = f"DATA_{ds}" if ds else "DATA"
    sheet_pc = f"PRICE_COMPARE_{ds}" if ds else "PRICE_COMPARE"

    xl = pd.ExcelFile(xlsx_path)
    sheets = set(xl.sheet_names)

    if sheet_data not in sheets:
        if "DATA" in sheets:
            sheet_data = "DATA"
        else:
            raise ValueError(f"DATA 시트를 찾을 수 없습니다: {xlsx_path}")

    if sheet_pc not in sheets:
        if "PRICE_COMPARE" in sheets:
            sheet_pc = "PRICE_COMPARE"
        else:
            raise ValueError(f"PRICE_COMPARE 시트를 찾을 수 없습니다: {xlsx_path}")

    df_data = pd.read_excel(xlsx_path, sheet_name=sheet_data)
    df_pc = pd.read_excel(xlsx_path, sheet_name=sheet_pc)

    required_data = ["batch_date", "mall_name", "manufacturer", "product_key", "product_name"]
    required_pc = ["batch_date", "mall_name", "manufacturer", "product_key", "product_name"]

    for col in required_data:
        if col not in df_data.columns:
            raise ValueError(f"DATA 시트에 필수 컬럼이 없습니다: {col}")
    for col in required_pc:
        if col not in df_pc.columns:
            raise ValueError(f"PRICE_COMPARE 시트에 필수 컬럼이 없습니다: {col}")

    for df in (df_data, df_pc):
        df["batch_date"] = normalize_batch_date_series(df["batch_date"])
        for c in ["mall_name", "manufacturer", "product_key", "product_name"]:
            df[c] = df[c].astype(str)

    if "price" in df_data.columns:
        df_data["price"] = pd.to_numeric(df_data["price"], errors="coerce")
    for c in ["min_price", "avg_price", "max_price"]:
        if c in df_pc.columns:
            df_pc[c] = pd.to_numeric(df_pc[c], errors="coerce")

    df_data = df_data.dropna(subset=required_data).copy()
    df_pc = df_pc.dropna(subset=required_pc).copy()

    if ds:
        df_data["batch_date"] = ds
        df_pc["batch_date"] = ds

    return df_data, df_pc


def load_excels_multi(folder: str, filenames: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_list: List[pd.DataFrame] = []
    pc_list: List[pd.DataFrame] = []

    for fn in filenames:
        path = os.path.join(folder, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(f"엑셀 파일이 없습니다: {path}")
        dfd, dfp = load_excel_one(path)
        data_list.append(dfd)
        pc_list.append(dfp)

    df_data_all = pd.concat(data_list, ignore_index=True) if data_list else pd.DataFrame()
    df_pc_all = pd.concat(pc_list, ignore_index=True) if pc_list else pd.DataFrame()
    return df_data_all, df_pc_all


def discover_xlsx_files(folder: str, regex: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"폴더가 없습니다: {folder}")
    pat = re.compile(regex)
    files = []
    for name in os.listdir(folder):
        if pat.search(name) and name.lower().endswith(".xlsx"):
            files.append(name)
    files.sort()
    return files


# ============================================================
# Catalog
# ============================================================
def build_catalog(df_pc: pd.DataFrame, df_data: pd.DataFrame) -> Dict[str, Any]:
    malls = sorted(set(pd.concat([df_pc["mall_name"], df_data["mall_name"]]).astype(str)))
    manufacturers = sorted(set(pd.concat([df_pc["manufacturer"], df_data["manufacturer"]]).astype(str)))
    product_keys = sorted(set(pd.concat([df_pc["product_key"], df_data["product_key"]]).astype(str)))

    df_prod = (
        pd.concat([
            df_pc[["product_key", "product_name", "manufacturer"]],
            df_data[["product_key", "product_name", "manufacturer"]],
        ], ignore_index=True)
        .astype(str)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    name_to_key: Dict[str, str] = {}
    norm_name_to_key: Dict[str, str] = {}
    key_to_manufacturer: Dict[str, str] = {}

    product_names_sorted: List[str] = []

    for _, r in df_prod.iterrows():
        pk = str(r["product_key"])
        pn = str(r["product_name"])
        mn = str(r["manufacturer"])
        key_to_manufacturer.setdefault(pk, mn)
        name_to_key.setdefault(pn, pk)
        norm = normalize_lookup_text(pn)
        if norm:
            norm_name_to_key.setdefault(norm, pk)

    product_names_sorted = sorted(name_to_key.keys(), key=len, reverse=True)

    return {
        "malls": malls,
        "manufacturers": manufacturers,
        "product_key_set": set(product_keys),
        "name_to_key": name_to_key,
        "norm_name_to_key": norm_name_to_key,
        "key_to_manufacturer": key_to_manufacturer,
        "product_names_sorted": product_names_sorted,
    }


# ============================================================
# Parsing helpers
# ============================================================
def extract_target_mall(question: str, malls: List[str]) -> Optional[str]:
    q = question.strip()
    for m in sorted(malls, key=len, reverse=True):
        if m and m in q:
            return m
    return None


def extract_all_malls(question: str, malls: List[str]) -> List[str]:
    q = question.strip()
    hits: List[str] = []
    for m in sorted(malls, key=len, reverse=True):
        if m and m in q and m not in hits:
            hits.append(m)
    return hits


def parse_manufacturer(question: str, manufacturers: List[str]) -> Optional[str]:
    q = question.strip()
    for man in sorted(manufacturers, key=len, reverse=True):
        if man and man in q:
            return man

    m = re.search(r"\b([A-Za-z가-힣0-9]+)\s*(사|회사)\b", q)
    if m:
        head = m.group(1)
        cand = f"{head}제과"
        if cand in manufacturers:
            return cand
    return None


def parse_all_manufacturers(question: str, manufacturers: List[str]) -> List[str]:
    q = question.strip()
    hits: List[str] = []
    for man in sorted(manufacturers, key=len, reverse=True):
        if man and man in q and man not in hits:
            hits.append(man)

    alias_map = {
        "해태": "해태제과",
    }
    for alias, full in alias_map.items():
        if alias in q and full in manufacturers and full not in hits:
            hits.append(full)

    return hits


def resolve_my_manufacturer(question: str, manufacturers: List[str], fallback_my: str = MY_MANUFACTURER) -> Optional[str]:
    if has_my_words(question):
        return fallback_my
    return None


def parse_product_key(question: str, catalog: Dict[str, Any]) -> Optional[str]:
    q = question.strip()
    q_norm = normalize_lookup_text(q)
    keyset = catalog["product_key_set"]
    name_to_key: Dict[str, str] = catalog["name_to_key"]
    norm_name_to_key: Dict[str, str] = catalog.get("norm_name_to_key", {})
    product_names_sorted: List[str] = catalog.get("product_names_sorted", [])

    for pname in product_names_sorted:
        if pname and pname in q:
            pk = name_to_key.get(pname)
            if pk in keyset:
                return pk

    if q_norm:
        for norm_name, pk in sorted(norm_name_to_key.items(), key=lambda x: len(x[0]), reverse=True):
            if norm_name and norm_name in q_norm and pk in keyset:
                return pk

    candidates: List[str] = []
    for m in re.finditer(r"\b([A-Za-z])\s*[-_]?\s*0*([0-9]{1,4})\b", q):
        prefix = m.group(1).upper()
        num = int(m.group(2))
        candidates.append(f"{prefix}{num:03d}")

    m2 = re.search(r"([A-Za-z가-힣]+)\s*과자\s*[-_]?\s*0*([0-9]{1,4})", q)
    if m2:
        head = m2.group(1)
        num = int(m2.group(2))
        cand_name = f"{head}과자{num:03d}"
        if cand_name in name_to_key:
            pk = name_to_key[cand_name]
            if pk in keyset:
                return pk

        cand_norm = normalize_lookup_text(cand_name)
        if cand_norm in norm_name_to_key:
            pk = norm_name_to_key[cand_norm]
            if pk in keyset:
                return pk

    for cand in candidates:
        if cand in keyset:
            return cand

    return None


def looks_like_product_query(q: str) -> bool:
    q_norm = normalize_lookup_text(q)
    return (
        bool(re.search(r"\b[A-Za-z]\s*[-_]?\s*\d{1,4}\b", q))
        or bool(re.search(r"[가-힣A-Za-z]+\s*과자\s*[-_]?\s*0*\d{1,4}", q))
        or bool(re.search(r"[A-Z가-힣]+\d{2,4}", q_norm))
    )


# ============================================================
# Date parsing
# ============================================================
def parse_batch_date(question: str, default_date: str) -> str:
    q = normalize_range_separators(question.strip())

    mrel = re.search(r"(\d{1,3})\s*일\s*(전|이전|앞)", q)
    if mrel:
        n = max(1, min(int(mrel.group(1)), 365))
        return shift_date_ymd(default_date, -n)

    mrel2 = re.search(r"(\d{1,3})\s*일\s*(후|뒤)", q)
    if mrel2:
        n = max(1, min(int(mrel2.group(1)), 365))
        return shift_date_ymd(default_date, +n)

    if "그제" in q or "그저께" in q:
        return shift_date_ymd(default_date, -2)
    if "어제" in q:
        return shift_date_ymd(default_date, -1)
    if "오늘" in q or "금일" in q:
        return default_date
    if "내일" in q:
        return shift_date_ymd(default_date, +1)
    if "모레" in q:
        return shift_date_ymd(default_date, +2)

    m = re.search(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", q)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"

    m2 = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", q)
    if m2:
        y, mo, d = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"

    try:
        inferred_year = int(str(default_date)[:4])
    except Exception:
        inferred_year = datetime.now().year

    mm = re.search(r"(?<!\d)(\d{1,2})[-/](\d{1,2})(?!\d)", q)
    if mm:
        mo, d = int(mm.group(1)), int(mm.group(2))
        return f"{inferred_year:04d}-{mo:02d}-{d:02d}"

    return default_date


def parse_dates_list(question: str, default_date: str) -> List[str]:
    q = normalize_range_separators(question.strip())
    if not q:
        return []

    rel_map = [
        ("그저께", -2),
        ("그제", -2),
        ("어제", -1),
        ("오늘", 0),
        ("금일", 0),
        ("내일", +1),
        ("모레", +2),
    ]
    rel_hits: List[Tuple[int, str]] = []
    for kw, delta in rel_map:
        idx = q.find(kw)
        if idx >= 0:
            rel_hits.append((idx, shift_date_ymd(default_date, delta)))

    for mm in re.finditer(r"(\d{1,3})\s*일\s*(전|이전|앞)", q):
        n = max(1, min(int(mm.group(1)), 365))
        rel_hits.append((mm.start(), shift_date_ymd(default_date, -n)))
    for mm in re.finditer(r"(\d{1,3})\s*일\s*(후|뒤)", q):
        n = max(1, min(int(mm.group(1)), 365))
        rel_hits.append((mm.start(), shift_date_ymd(default_date, +n)))

    if rel_hits:
        rel_hits.sort(key=lambda x: x[0])
        seen = set()
        out: List[str] = []
        for _, d in rel_hits:
            if d not in seen:
                seen.add(d)
                out.append(d)
        return out

    dates: List[Tuple[int, str]] = []

    for mm in re.finditer(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", q):
        y, mo, d = int(mm.group(1)), int(mm.group(2)), int(mm.group(3))
        dates.append((mm.start(), f"{y:04d}-{mo:02d}-{d:02d}"))

    for mm in re.finditer(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", q):
        y, mo, d = int(mm.group(1)), int(mm.group(2)), int(mm.group(3))
        dates.append((mm.start(), f"{y:04d}-{mo:02d}-{d:02d}"))

    try:
        inferred_year = int(str(default_date)[:4])
    except Exception:
        inferred_year = datetime.now().year

    for mm in re.finditer(r"(?<!\d)(\d{1,2})[-/](\d{1,2})(?!\d)", q):
        left = q[max(0, mm.start() - 5):mm.start()]
        if re.search(r"\d{4}$", left):
            continue
        mo, d = int(mm.group(1)), int(mm.group(2))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            dates.append((mm.start(), f"{inferred_year:04d}-{mo:02d}-{d:02d}"))

    dates.sort(key=lambda x: x[0])

    seen = set()
    uniq: List[str] = []
    for _, d in dates:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def has_explicit_range_pattern(q: str) -> bool:
    if "~" in q:
        return True
    if ("부터" in q and "까지" in q):
        return True
    if re.search(r"\d{4}-\d{1,2}-\d{1,2}\s*-\s*\d{4}-\d{1,2}-\d{1,2}", q):
        return True
    if re.search(r"\d{1,2}/\d{1,2}\s*-\s*\d{1,2}/\d{1,2}", q):
        return True
    if re.search(r"\d{1,2}-\d{1,2}\s*-\s*\d{1,2}-\d{1,2}", q):
        return True
    return False


def resolve_dates(question: str, default_end: str, default_start: str) -> DateResolution:
    q = normalize_range_separators(question.strip())
    requested_start = ""
    requested_end = ""
    resolved_start = default_end
    resolved_end = default_end
    source = "single_date"

    try:
        dt_end = datetime.strptime(str(default_end)[:10], "%Y-%m-%d")
    except Exception:
        dt_end = datetime.now()

    if any(k in q for k in ["이번주", "이번 주", "금주"]):
        s, e = _week_range_for(dt_end)
        requested_start, requested_end = s, e
        source = "calendar_range"
    elif any(k in q for k in ["지난주", "지난 주", "저번주", "저번 주", "전주"]):
        base = dt_end - timedelta(days=7)
        s, e = _week_range_for(base)
        requested_start, requested_end = s, e
        source = "calendar_range"
    elif any(k in q for k in ["이번달", "이번 달", "금월"]):
        s, e = _month_range_for(dt_end)
        requested_start, requested_end = s, e
        source = "calendar_range"
    elif any(k in q for k in ["지난달", "지난 달", "저번달", "저번 달", "전월"]):
        y, m = dt_end.year, dt_end.month
        if m == 1:
            y, m = y - 1, 12
        else:
            m -= 1
        dt_prev = datetime(y, m, 15)
        s, e = _month_range_for(dt_prev)
        requested_start, requested_end = s, e
        source = "calendar_range"
    else:
        m = re.search(r"(최근|지난)\s*([0-9]{1,3})\s*일(간|동안|일간)?", q)
        if m:
            n = max(1, min(int(m.group(2)), 365))
            requested_end = default_end
            requested_start = (datetime.strptime(default_end, "%Y-%m-%d") - timedelta(days=n - 1)).strftime("%Y-%m-%d")
            source = "relative_range"
        else:
            dl = parse_dates_list(q, default_date=default_end)
            if len(dl) >= 2 and has_explicit_range_pattern(q):
                requested_start, requested_end = dl[0], dl[1]
                source = "explicit_range"
            else:
                single = parse_batch_date(q, default_date=default_end)
                requested_start = single
                requested_end = single
                source = "single_date"

    if requested_start > requested_end:
        requested_start, requested_end = requested_end, requested_start

    clamped_start = requested_start < default_start
    clamped_end = requested_end > default_end

    resolved_start = max(requested_start, default_start)
    resolved_end = min(requested_end, default_end)

    is_range = resolved_start != resolved_end

    return DateResolution(
        requested_start=requested_start,
        requested_end=requested_end,
        resolved_start=resolved_start,
        resolved_end=resolved_end,
        is_range=is_range,
        clamped_start=clamped_start,
        clamped_end=clamped_end,
        source=source,
    )


def parse_top_n(question: str, default_n: int = 10) -> int:
    q = normalize_range_separators(question.strip())

    m = re.search(r"상위\s*([0-9]{1,4})\s*개", q)
    if m:
        return int(m.group(1))

    m2 = re.search(r"(탑|TOP|top|베스트|랭킹|순위)\s*[-]?\s*([0-9]{1,4})", q)
    if m2:
        return int(m2.group(2))

    m3 = re.search(r"\btop\s*[-]?\s*([0-9]{1,4})\b", q, flags=re.IGNORECASE)
    if m3:
        return int(m3.group(1))

    m4 = re.search(r"([0-9]{1,4})\s*개", q)
    if m4:
        return int(m4.group(1))

    return int(default_n)


# ============================================================
# Semantic helpers
# ============================================================
def wants_target_price(q: str) -> bool:
    return any(k in q for k in ["가격과 함께", "가격도", "가격 함께", "같이", "함께", "가격까지", "타겟몰 가격", "가격 포함"])


def wants_diff(q: str) -> bool:
    return any(k in q for k in ["차액", "차이", "diff", "얼마나 더", "더 비싸", "비교", "갭", "gap"])


def wants_date_in_rows(q: str) -> bool:
    return any(k in q for k in [
        "날짜와 함께", "해당 날짜와 함께", "일자와 함께",
        "날짜 포함", "일자 포함", "날짜별", "일자별",
        "날짜별로", "일자별로"
    ])


def prefers_summary_semantic(q: str) -> bool:
    return any(k in q for k in [
        "요약", "요약 가격표", "가격표", "summary",
        "전체 데이터 말고", "원본 말고", "raw 말고", "요약표"
    ])


def prefers_raw_semantic(q: str) -> bool:
    return any(k in q for k in [
        "원본", "raw", "오퍼", "오퍼 목록", "전건", "원자료", "상세"
    ])


def wants_all_rows(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ["전부", "전체", "모두", "all"])


def wants_group_by_mall(q: str) -> bool:
    return any(k in q for k in ["몰별", "쇼핑몰별", "몰마다", "쇼핑몰마다", "묶어서"])


def has_generic_price_bucket_words(q: str) -> bool:
    return any(k in q for k in [
        "싼 것들", "저렴한 것들", "비싼 것들", "높은 것들", "낮은 것들",
        "싼 상품", "저렴한 상품", "비싼 상품", "높은 상품", "낮은 상품",
        "싼 애들", "비싼 애들", "저렴한 애들",
        "싼 제품", "비싼 제품", "높은 애들", "낮은 애들"
    ])


def has_compare_words(q: str) -> bool:
    return any(k in q for k in COMPARE_WORDS)


def raw_limit_for_question(q: str) -> int:
    if wants_all_rows(q):
        return HARD_LIMIT_ROWS
    return DEFAULT_RAW_LIMIT


def summary_limit_for_question(q: str) -> int:
    if wants_all_rows(q):
        return HARD_LIMIT_ROWS
    n = parse_top_n(q, default_n=DEFAULT_SUMMARY_LIMIT)
    return max(1, min(HARD_LIMIT_ROWS, n))


def parse_metric_kor(question: str) -> str:
    q = question.strip()
    ql = q.lower()

    if "min_price" in q:
        return "min_price"
    if "avg_price" in q:
        return "avg_price"
    if "max_price" in q:
        return "max_price"

    if ("평균" in q) or ("avg" in ql) or ("평균가" in q) or ("시세" in q) or ("평균값" in q):
        return "avg_price"

    if any(k in q for k in ["최저가", "최저", "저렴", "싼", "싸게", "제일 싼", "가장 싼", "가장 저렴", "낮은"]):
        return "min_price"

    if any(k in q for k in ["최고가", "최대", "비싼", "높은", "큰값", "제일 비싼", "가장 비싼", "평균값 큰"]):
        return "max_price"

    if has_compare_words(q):
        return "avg_price"

    return "min_price"


def parse_sort_direction(question: str, metric: str) -> bool:
    q = question.strip()
    if any(k in q for k in ["높", "비싼", "최고", "큰", "비싸게", "가장 비싼", "높은 순", "내림차순", "더 비싼"]):
        return False
    if any(k in q for k in ["낮", "저렴", "싼", "낮은 순", "오름차순", "더 싼"]):
        return True
    if metric == "max_price":
        return False
    return True


def _has_violation_semantics(q: str) -> bool:
    return any(w in q for w in VIOLATION_WORDS)


def _has_trend_semantics(q: str) -> bool:
    return any(k in q for k in TREND_WORDS_EXT)


def _has_detail_semantics(q: str) -> bool:
    return any(k in q for k in DETAIL_WORDS_EXT)


def _has_offer_semantics(q: str) -> bool:
    return any(k in q for k in OFFER_WORDS)


def _has_price_semantics(q: str) -> bool:
    return any(w in q for w in PRICE_WORDS_EXT)


def _has_topn_semantics(q: str) -> bool:
    if any(w in q for w in TOP_WORDS_EXT):
        return True
    if re.search(r"([0-9]{1,4})\s*개", q):
        return True
    if any(k in q for k in ["높은 상품", "낮은 상품", "싼 상품", "비싼 상품", "싼 것들", "비싼 것들"]):
        return True
    return False


def _has_shipping_issue_semantics(q: str) -> bool:
    has_ship = any(w in q for w in SHIP_WORDS)
    has_neg = any(w in q for w in SHIP_NEG_WORDS)
    return has_ship and has_neg


def _has_metric_semantics(q: str) -> bool:
    return any(k in q for k in [
        "min_price", "avg_price", "max_price",
        "최저가", "최고가", "평균", "최대", "최저",
        "저렴", "비싼", "높은", "제일 싼", "가장 싼", "가장 비싼",
        "싼 것들", "비싼 것들", "높은 것들", "낮은 것들", "시세", "비싼 쪽", "싼 쪽"
    ])


def _has_range_semantics(q: str) -> bool:
    if any(k in q for k in [
        "부터", "까지", "최근", "지난",
        "이번주", "이번 주", "지난주", "지난 주",
        "이번달", "이번 달", "지난달", "지난 달",
        "금주", "금월", "전월", "일주일"
    ]):
        return True
    return has_explicit_range_pattern(q)


def _is_not_cheapest_semantic(q: str) -> bool:
    has_price_compare = any(k in q for k in [
        "최저가", "가장 싸", "가장 저렴", "싼", "저렴", "더 싸", "더 저렴", "제일 싼", "놓친"
    ])
    return (
        ("최저가" in q and any(k in q for k in ["아닌", "아니다", "아닙", "놓친", "못한", "실패"]))
        or (("다른" in q or "다른곳" in q or "다른 곳" in q or "타" in q)
            and any(k in q for k in ["보다", "더"])
            and any(k in q for k in ["저렴", "싼", "싸", "낮"]))
        or (("보다" in q)
            and any(k in q for k in ["싼", "저렴", "싸", "더 낮"])
            and any(k in q for k in ["곳", "몰", "사이트"]))
        or (("가장" in q or "제일" in q)
            and any(k in q for k in ["싸지", "저렴하지"])
            and any(k in q for k in ["않", "아니"]))
        or (has_price_compare and any(k in q for k in ["아닌", "아니다", "아닙", "놓친"]))
    )


def _is_cheapest_only_semantic(q: str) -> bool:
    positive_phrase = any(k in q for k in [
        "최저가",
        "최저가인",
        "제일 싼",
        "가장 싼",
        "가장 저렴",
        "가장 저렴한",
        "제일 저렴",
        "제일 저렴한",
    ])
    item_phrase = any(k in q for k in ["상품", "제품", "과자", "것", "품목"])
    mall_subject = any(k in q for k in ["이 최저가", "이(가) 최저가", "이 가장 싼", "이(가) 가장 싼", "이 가장 저렴", "이(가) 가장 저렴"])

    return (
        positive_phrase
        and item_phrase
        and ("아닌" not in q)
        and ("놓친" not in q)
        and (("만" in q) or mall_subject or ("최저가인 상품" in q) or ("가장 싼 상품" in q) or ("가장 저렴한 상품" in q))
    )


def infer_view_source(q: str) -> str:
    if prefers_raw_semantic(q) and not prefers_summary_semantic(q):
        return "raw"
    if prefers_summary_semantic(q):
        return "summary"
    if has_generic_price_bucket_words(q):
        return "summary"
    if any(k in q for k in ["가격표", "요약표", "최저가", "평균가", "최고가", "추이", "시세"]):
        return "summary"
    if any(k in q for k in ["상세", "원본", "오퍼"]):
        return "raw"
    return "unknown"


def extract_semantic_slots(q: str) -> SemanticSlots:
    return SemanticSlots(
        has_violation=_has_violation_semantics(q),
        wants_detail=_has_detail_semantics(q),
        wants_trend=_has_trend_semantics(q),
        wants_offer=_has_offer_semantics(q),
        wants_price=_has_price_semantics(q),
        wants_topn=_has_topn_semantics(q),
        wants_shipping_issue=_has_shipping_issue_semantics(q),
        has_metric=_has_metric_semantics(q),
        has_range_expr=_has_range_semantics(q),
        not_cheapest=_is_not_cheapest_semantic(q),
        cheapest_only=_is_cheapest_only_semantic(q),
        wants_date_in_rows=wants_date_in_rows(q),
        prefers_summary=prefers_summary_semantic(q),
        prefers_raw=prefers_raw_semantic(q),
        wants_all_rows=wants_all_rows(q),
        wants_group_by_mall=wants_group_by_mall(q),
        has_generic_price_bucket_words=has_generic_price_bucket_words(q),
        has_compare_words=has_compare_words(q),
        view_source=infer_view_source(q),
    )


# ============================================================
# Intent routing
# ============================================================
IntentRuleFn = Callable[[ParsedEntities, DateResolution, SemanticSlots], bool]


@dataclass
class IntentRule:
    name: str
    predicate: IntentRuleFn


def build_intent_rules() -> List[IntentRule]:
    return [
        IntentRule("A1_VIOL_TREND", lambda e, d, s:
                   bool(e.mall) and s.has_violation and (s.wants_trend or d.is_range)),

        IntentRule("A1_VIOL_DETAIL", lambda e, d, s:
                   bool(e.mall) and s.has_violation),

        IntentRule("Q1", lambda e, d, s:
                   bool(e.product_key) and s.wants_price and not (s.wants_offer or s.prefers_raw) and not s.wants_trend),

        IntentRule("Q1_DETAIL", lambda e, d, s:
                   bool(e.product_key) and (s.wants_offer or s.prefers_raw)),

        IntentRule("Q1_DETAIL", lambda e, d, s:
                   bool(e.product_key) and s.wants_detail and not s.wants_price),

        IntentRule("Q5_RANGE_DETAIL", lambda e, d, s:
                   bool(e.mall) and s.not_cheapest and d.is_range and (s.wants_detail or s.wants_all_rows or any(k in s.view_source for k in ["raw"]))),

        IntentRule("Q5_RANGE_SUMMARY", lambda e, d, s:
                   bool(e.mall) and s.not_cheapest and d.is_range),

        IntentRule("Q5", lambda e, d, s:
                   bool(e.mall) and s.not_cheapest and not d.is_range),

        IntentRule("Q6", lambda e, d, s:
                   bool(e.mall) and s.cheapest_only and not d.is_range),

        IntentRule("Q7", lambda e, d, s:
                   bool(e.mall) and s.wants_topn and s.has_metric and not d.is_range and not bool(e.product_key)),

        IntentRule("Q8_TREND", lambda e, d, s:
                   bool(e.product_key) and s.wants_trend),

        IntentRule("Q4", lambda e, d, s:
                   s.wants_shipping_issue and not s.has_compare_words),

        IntentRule("Q7D_MANU_RANGE_TOPN_BY_MALL", lambda e, d, s:
                   bool(e.manufacturer) and d.is_range and s.has_metric and s.wants_group_by_mall and not bool(e.product_key)),

        IntentRule("Q7C_MANU_RANGE_TOPN_WITH_DATE", lambda e, d, s:
                   bool(e.manufacturer) and d.is_range and s.has_metric and s.wants_date_in_rows and not bool(e.product_key)),

        IntentRule("Q7B_MANU_RANGE_TOPN", lambda e, d, s:
                   bool(e.manufacturer) and d.is_range and s.has_metric and not bool(e.product_key) and not bool(e.mall)),

        IntentRule("Q9_MALL_SUMMARY_TABLE", lambda e, d, s:
                   bool(e.mall) and s.prefers_summary and not bool(e.product_key)),

        IntentRule("Q1", lambda e, d, s:
                   bool(e.product_key) and not s.wants_detail),
    ]


def detect_intent(entities: ParsedEntities, dates: DateResolution, slots: SemanticSlots) -> str:
    if slots.has_compare_words:
        return "UNKNOWN"

    for rule in build_intent_rules():
        try:
            if rule.predicate(entities, dates, slots):
                return rule.name
        except Exception:
            continue
    return "UNKNOWN"


# ============================================================
# Structured queries
# ============================================================
def q1_product_best(df_pc: pd.DataFrame, product_key: str, batch_date: str):
    cur = df_pc[(df_pc["batch_date"] == batch_date) & (df_pc["product_key"] == product_key)].copy()
    if len(cur) == 0:
        return None
    best = cur.loc[cur["min_price"].idxmin()]
    table = cur[["manufacturer", "mall_name", "min_price", "avg_price", "max_price"]].sort_values("min_price")
    return {
        "manufacturer": str(best.get("manufacturer", "")),
        "best_mall": str(best["mall_name"]),
        "best_price": int(best["min_price"]),
        "table": table,
    }


def q1_product_all_offers(df_data: pd.DataFrame, product_key: str, batch_date: str) -> Optional[pd.DataFrame]:
    cur = df_data[(df_data["batch_date"] == batch_date) & (df_data["product_key"] == product_key)].copy()
    if len(cur) == 0:
        return None

    cols = [c for c in DETAIL_RAW_DEFAULT_SELECT if c in cur.columns]
    sort_cols = []
    if "mall_name" in cur.columns:
        sort_cols.append("mall_name")
    if "rank" in cur.columns:
        sort_cols.append("rank")
    if not sort_cols and "date" in cur.columns:
        sort_cols = ["date"]

    if sort_cols:
        return cur[cols].sort_values(sort_cols, ascending=True).reset_index(drop=True)
    return cur[cols].reset_index(drop=True)


def q4_shipping_issues(df_data_all_days: pd.DataFrame, start_date: str, end_date: str, product_key: Optional[str] = None):
    cur = df_data_all_days[
        (df_data_all_days["batch_date"] >= start_date) &
        (df_data_all_days["batch_date"] <= end_date)
    ].copy()
    if product_key:
        cur = cur[cur["product_key"] == product_key].copy()
    if len(cur) == 0 or "comments_top5" not in cur.columns:
        return None

    cur["ship_cnt"] = cur["comments_top5"].astype(str).apply(ship_issue_count)
    grp = cur.groupby("mall_name").agg(
        ship_issue_mentions=("ship_cnt", "sum"),
        rows_count=("ship_cnt", "count"),
    ).reset_index()
    grp["rate"] = grp["ship_issue_mentions"] / grp["rows_count"]
    return grp.sort_values(["rate", "ship_issue_mentions"], ascending=False)


def q5_mall_not_cheapest(
    df_pc: pd.DataFrame,
    batch_date: str,
    target_mall: str,
    manufacturer: Optional[str] = None,
    include_target_price: bool = False,
    include_diff: bool = False,
    require_target_price: bool = False,
):
    cur = df_pc[df_pc["batch_date"] == batch_date].copy()
    if manufacturer:
        cur = cur[cur["manufacturer"] == manufacturer].copy()
    if len(cur) == 0:
        return None

    tgt = cur[cur["mall_name"] == target_mall][["manufacturer", "product_key", "product_name", "min_price"]].copy()
    tgt = tgt.rename(columns={"min_price": "target_mall_price"})

    if len(tgt) == 0:
        return {
            "batch_date": batch_date,
            "target_mall": target_mall,
            "manufacturer": manufacturer,
            "total_products": 0,
            "product_count": 0,
            "case_count": 0,
            "table": pd.DataFrame(),
        }

    others = cur[cur["mall_name"] != target_mall][["manufacturer", "product_key", "product_name", "mall_name", "min_price"]].copy()
    others = others.rename(columns={"mall_name": "cheaper_mall", "min_price": "cheaper_price"})

    out = others.merge(tgt[["product_key", "target_mall_price"]], on="product_key", how="inner")
    out = out[out["cheaper_price"] < out["target_mall_price"]].copy()

    if require_target_price:
        out = out[pd.notna(out["target_mall_price"])].copy()

    if include_diff:
        out["diff"] = out["target_mall_price"] - out["cheaper_price"]

    base_cols = ["manufacturer", "product_key", "product_name", "cheaper_mall", "cheaper_price"]
    extra_cols: List[str] = []
    if include_target_price or include_diff or require_target_price:
        extra_cols.append("target_mall_price")
    if include_diff:
        extra_cols.append("diff")

    out = out[base_cols + extra_cols].copy()
    out = out.sort_values(["product_key", "cheaper_price", "cheaper_mall"], ascending=True).reset_index(drop=True)

    return {
        "batch_date": batch_date,
        "target_mall": target_mall,
        "manufacturer": manufacturer,
        "total_products": int(tgt["product_key"].nunique()),
        "product_count": int(out["product_key"].nunique()),
        "case_count": int(len(out)),
        "table": out,
    }


def q5_range_summary(
    df_pc: pd.DataFrame,
    start_date: str,
    end_date: str,
    target_mall: str,
    manufacturer: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    cur = df_pc[(df_pc["batch_date"] >= start_date) & (df_pc["batch_date"] <= end_date)].copy()
    if manufacturer:
        cur = cur[cur["manufacturer"] == manufacturer].copy()
    if len(cur) == 0:
        return None

    dates = sorted(cur["batch_date"].astype(str).unique().tolist())
    rows: List[Dict[str, Any]] = []

    for d in dates:
        day = cur[cur["batch_date"] == d].copy()
        if len(day) == 0:
            continue

        tgt = day[day["mall_name"] == target_mall][["product_key", "min_price"]].copy()
        tgt = tgt.rename(columns={"min_price": "target_mall_price"})

        if len(tgt) == 0:
            continue

        others = day[day["mall_name"] != target_mall][["manufacturer", "product_key", "mall_name", "min_price"]].copy()
        others = others.rename(columns={"mall_name": "cheaper_mall", "min_price": "cheaper_price"})

        merged = others.merge(tgt, on="product_key", how="inner")
        merged = merged[merged["cheaper_price"] < merged["target_mall_price"]].copy()

        total_products = int(tgt["product_key"].nunique())
        if len(merged) == 0:
            continue

        grp = (
            merged.groupby(["cheaper_mall"], dropna=False)["product_key"]
            .nunique()
            .reset_index(name="violations_count")
            .sort_values(["violations_count", "cheaper_mall"], ascending=[False, True])
            .reset_index(drop=True)
        )

        for _, r in grp.iterrows():
            row = {
                "batch_date": d,
                "target_mall": target_mall,
                "cheaper_mall": str(r["cheaper_mall"]),
                "violations_count": int(r["violations_count"]),
                "total_products": total_products,
            }
            if manufacturer:
                row["manufacturer"] = manufacturer
            rows.append(row)

    if not rows:
        return None

    out = pd.DataFrame(rows)
    ordered_cols = ["batch_date"]
    if "manufacturer" in out.columns:
        ordered_cols.append("manufacturer")
    ordered_cols += ["target_mall", "cheaper_mall", "violations_count", "total_products"]

    out = out[ordered_cols].sort_values(
        ["batch_date", "violations_count", "cheaper_mall"],
        ascending=[True, False, True]
    ).reset_index(drop=True)
    return out


def q5_range_detail(
    df_pc: pd.DataFrame,
    start_date: str,
    end_date: str,
    target_mall: str,
    manufacturer: Optional[str] = None,
    include_target_price: bool = True,
    include_diff: bool = True,
    limit: int = HARD_LIMIT_ROWS,
) -> Optional[pd.DataFrame]:
    cur = df_pc[(df_pc["batch_date"] >= start_date) & (df_pc["batch_date"] <= end_date)].copy()
    if manufacturer:
        cur = cur[cur["manufacturer"] == manufacturer].copy()
    if len(cur) == 0:
        return None

    tgt = cur[cur["mall_name"] == target_mall][["batch_date", "manufacturer", "product_key", "product_name", "min_price"]].copy()
    tgt = tgt.rename(columns={"min_price": "target_mall_price"})

    others = cur[cur["mall_name"] != target_mall][["batch_date", "manufacturer", "product_key", "product_name", "mall_name", "min_price"]].copy()
    others = others.rename(columns={"mall_name": "cheaper_mall", "min_price": "cheaper_price"})

    out = others.merge(
        tgt[["batch_date", "product_key", "product_name", "target_mall_price"]],
        on=["batch_date", "product_key", "product_name"],
        how="inner",
    )
    out = out[out["cheaper_price"] < out["target_mall_price"]].copy()
    if len(out) == 0:
        return pd.DataFrame(columns=[
            "batch_date", "manufacturer", "product_key", "product_name",
            "cheaper_mall", "cheaper_price", "target_mall_price", "diff"
        ])

    if include_diff:
        out["diff"] = out["target_mall_price"] - out["cheaper_price"]

    keep = ["batch_date", "manufacturer", "product_key", "product_name", "cheaper_mall", "cheaper_price"]
    if include_target_price:
        keep.append("target_mall_price")
    if include_diff:
        keep.append("diff")

    out = out[keep].sort_values(
        ["batch_date", "product_key", "cheaper_price", "cheaper_mall"],
        ascending=[True, True, True, True]
    ).head(int(max(1, min(limit, HARD_LIMIT_ROWS)))).reset_index(drop=True)
    return out


def q6_mall_is_cheapest(df_pc: pd.DataFrame, batch_date: str, target_mall: str, manufacturer: Optional[str] = None):
    cur = df_pc[df_pc["batch_date"] == batch_date].copy()
    if manufacturer:
        cur = cur[cur["manufacturer"] == manufacturer].copy()
    if len(cur) == 0:
        return None

    min_per_product = cur.groupby("product_key")["min_price"].transform("min")
    best_rows = cur[cur["min_price"] == min_per_product].copy()

    hit = best_rows[best_rows["mall_name"] == target_mall].copy()
    out = hit[["manufacturer", "product_key", "product_name", "mall_name", "min_price"]].rename(
        columns={"mall_name": "cheapest_mall", "min_price": "cheapest_price"}
    ).sort_values(["product_key", "cheapest_mall"], ascending=True).reset_index(drop=True)

    return {
        "batch_date": batch_date,
        "target_mall": target_mall,
        "manufacturer": manufacturer,
        "total_products": int(cur["product_key"].nunique()),
        "product_count": int(out["product_key"].nunique()),
        "case_count": int(len(out)),
        "table": out,
    }


def q7_mall_metric_topn(
    df_pc: pd.DataFrame,
    batch_date: str,
    target_mall: str,
    metric: str,
    n: int,
    ascending: bool,
    manufacturer: Optional[str] = None,
):
    cur = df_pc[(df_pc["batch_date"] == batch_date) & (df_pc["mall_name"] == target_mall)].copy()
    if manufacturer:
        cur = cur[cur["manufacturer"] == manufacturer].copy()
    if len(cur) == 0 or metric not in cur.columns:
        return None

    cur = cur.dropna(subset=[metric]).copy()
    cur[metric] = pd.to_numeric(cur[metric], errors="coerce")
    cur = cur.dropna(subset=[metric]).copy()

    out = cur[["manufacturer", "mall_name", "product_key", "product_name", metric]].sort_values(
        metric, ascending=bool(ascending)
    ).head(int(max(1, n))).copy()
    out = out.rename(columns={metric: "value"})
    return {
        "batch_date": batch_date,
        "target_mall": target_mall,
        "manufacturer": manufacturer,
        "metric": metric,
        "n": int(n),
        "ascending": bool(ascending),
        "table": out,
    }


def q8_product_trend(
    df_pc: pd.DataFrame,
    product_key: str,
    start_date: str,
    end_date: str,
    manufacturer: Optional[str] = None,
    mall: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    cur = df_pc[
        (df_pc["product_key"] == product_key) &
        (df_pc["batch_date"] >= start_date) &
        (df_pc["batch_date"] <= end_date)
    ].copy()
    if manufacturer:
        cur = cur[cur["manufacturer"] == manufacturer].copy()
    if mall:
        cur = cur[cur["mall_name"] == mall].copy()
    if len(cur) == 0:
        return None
    return cur[["batch_date", "manufacturer", "mall_name", "min_price", "avg_price", "max_price"]].sort_values(
        ["batch_date", "mall_name"], ascending=[True, True]
    ).reset_index(drop=True)


def q7b_manufacturer_range_topn(
    df_pc: pd.DataFrame,
    start_date: str,
    end_date: str,
    manufacturer: str,
    metric: str,
    n: int,
    ascending: bool,
) -> Optional[pd.DataFrame]:
    cur = df_pc[
        (df_pc["batch_date"] >= start_date) &
        (df_pc["batch_date"] <= end_date) &
        (df_pc["manufacturer"] == manufacturer)
    ].copy()
    if len(cur) == 0 or metric not in cur.columns:
        return None

    grp = cur.groupby(["manufacturer", "mall_name", "product_key", "product_name"], dropna=False)[metric].mean().reset_index()
    grp = grp.rename(columns={metric: "value"})
    grp = grp.sort_values("value", ascending=ascending).head(int(max(1, n))).reset_index(drop=True)
    return grp


def q7c_manufacturer_range_topn_with_date(
    df_pc: pd.DataFrame,
    start_date: str,
    end_date: str,
    manufacturer: str,
    metric: str,
    n: int,
    ascending: bool,
) -> Optional[pd.DataFrame]:
    cur = df_pc[
        (df_pc["batch_date"] >= start_date) &
        (df_pc["batch_date"] <= end_date) &
        (df_pc["manufacturer"] == manufacturer)
    ].copy()
    if len(cur) == 0 or metric not in cur.columns:
        return None

    cols = ["batch_date", "manufacturer", "mall_name", "product_key", "product_name", metric]
    cur = cur[cols].rename(columns={metric: "value"})
    cur["value"] = pd.to_numeric(cur["value"], errors="coerce")
    cur = cur.dropna(subset=["value"]).copy()

    cur = cur.sort_values(["value", "batch_date", "mall_name", "product_key"], ascending=[ascending, True, True, True])
    cur = cur.head(int(max(1, n))).reset_index(drop=True)
    return cur


def q7d_manufacturer_range_topn_by_mall(
    df_pc: pd.DataFrame,
    start_date: str,
    end_date: str,
    manufacturer: str,
    metric: str,
    n_per_mall: int,
    ascending: bool,
) -> Optional[Dict[str, pd.DataFrame]]:
    cur = df_pc[
        (df_pc["batch_date"] >= start_date) &
        (df_pc["batch_date"] <= end_date) &
        (df_pc["manufacturer"] == manufacturer)
    ].copy()
    if len(cur) == 0 or metric not in cur.columns:
        return None

    grp = cur.groupby(["manufacturer", "mall_name", "product_key", "product_name"], dropna=False)[metric].mean().reset_index()
    grp = grp.rename(columns={metric: "value"})

    out: Dict[str, pd.DataFrame] = {}
    for mall_name, g in grp.groupby("mall_name", dropna=False):
        g2 = g.sort_values("value", ascending=ascending).head(int(max(1, n_per_mall))).reset_index(drop=True)
        out[str(mall_name)] = g2

    if not out:
        return None
    return out


def q9_mall_summary_table(
    df_pc: pd.DataFrame,
    start_date: str,
    end_date: str,
    target_mall: str,
    manufacturer: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    cur = df_pc[
        (df_pc["batch_date"] >= start_date) &
        (df_pc["batch_date"] <= end_date) &
        (df_pc["mall_name"] == target_mall)
    ].copy()
    if manufacturer:
        cur = cur[cur["manufacturer"] == manufacturer].copy()
    if len(cur) == 0:
        return None

    cols = ["batch_date", "mall_name", "manufacturer", "product_key", "product_name", "min_price", "avg_price", "max_price"]
    return cur[cols].sort_values(["batch_date", "product_key"], ascending=[True, True]).reset_index(drop=True)


def a1_violation_trend(
    df_pc: pd.DataFrame,
    start_date: str,
    end_date: str,
    manufacturer: str,
    target_mall: str,
) -> Optional[pd.DataFrame]:
    cur = df_pc[
        (df_pc["batch_date"] >= start_date) &
        (df_pc["batch_date"] <= end_date) &
        (df_pc["manufacturer"] == manufacturer)
    ].copy()
    if len(cur) == 0:
        return None

    gmin = cur.groupby(["batch_date", "product_key"], dropna=False)["min_price"].min().reset_index()
    gmin = gmin.rename(columns={"min_price": "global_min_price"})

    tgt = cur[cur["mall_name"] == target_mall][["batch_date", "product_key", "min_price"]].copy()
    tgt = tgt.rename(columns={"min_price": "target_mall_price"})

    merged = gmin.merge(tgt, on=["batch_date", "product_key"], how="left")
    merged = merged[pd.notna(merged["target_mall_price"])].copy()
    if len(merged) == 0:
        return None

    merged["diff"] = merged["target_mall_price"] - merged["global_min_price"]
    merged["is_violation"] = merged["diff"] > 0

    grp = merged.groupby("batch_date", dropna=False).agg(
        violations_count=("is_violation", "sum"),
        sum_diff=("diff", lambda x: float(x[x > 0].sum())),
        avg_diff=("diff", lambda x: float(x[x > 0].mean()) if (x > 0).any() else 0.0),
        total_products=("product_key", "nunique"),
    ).reset_index()

    grp["violations_count"] = grp["violations_count"].astype(int)
    grp["total_products"] = grp["total_products"].astype(int)
    return grp.sort_values("batch_date", ascending=True).reset_index(drop=True)


def a1_violation_detail(
    df_pc: pd.DataFrame,
    start_date: str,
    end_date: str,
    manufacturer: str,
    target_mall: str,
    limit: int = 5000,
) -> Optional[pd.DataFrame]:
    cur = df_pc[
        (df_pc["batch_date"] >= start_date) &
        (df_pc["batch_date"] <= end_date) &
        (df_pc["manufacturer"] == manufacturer)
    ].copy()
    if len(cur) == 0:
        return None

    tgt = cur[cur["mall_name"] == target_mall][["batch_date", "product_key", "product_name", "min_price"]].copy()
    tgt = tgt.rename(columns={"min_price": "target_mall_price"})

    others = cur[cur["mall_name"] != target_mall][["batch_date", "product_key", "product_name", "mall_name", "min_price"]].copy()
    others = others.rename(columns={"mall_name": "cheaper_mall", "min_price": "cheaper_price"})

    out = others.merge(tgt[["batch_date", "product_key", "target_mall_price"]], on=["batch_date", "product_key"], how="inner")
    out = out[out["cheaper_price"] < out["target_mall_price"]].copy()
    out["diff"] = out["target_mall_price"] - out["cheaper_price"]

    out = out[["batch_date", "product_key", "product_name", "target_mall_price", "cheaper_mall", "cheaper_price", "diff"]]
    out = out.sort_values(["batch_date", "product_key", "diff", "cheaper_price"], ascending=[True, True, False, True])
    out = out.head(int(max(1, min(limit, HARD_LIMIT_ROWS)))).reset_index(drop=True)
    return out


# ============================================================
# Chroma
# ============================================================
def make_id(batch_date: str, mall: str, pkey: str, chunk_type: str) -> str:
    return f"{batch_date}|{mall}|{pkey}|{chunk_type}"


def get_chroma_collections(chroma_path: str, coll_summary: str, coll_reviews: str):
    if chromadb is None or DefaultEmbeddingFunction is None:
        raise RuntimeError("chromadb가 설치되어 있지 않습니다.")
    client = chromadb.PersistentClient(path=chroma_path)
    ef = DefaultEmbeddingFunction()
    col_sum = client.get_or_create_collection(name=coll_summary, embedding_function=ef)
    col_rev = client.get_or_create_collection(name=coll_reviews, embedding_function=ef)
    return col_sum, col_rev


def upsert_price_compare(col_sum, df_pc: pd.DataFrame, batch_size: int = 500, version: str = "redesigned_v1"):
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    for _, r in df_pc.iterrows():
        batch_date = str(r["batch_date"])
        mall = str(r["mall_name"])
        manu = str(r.get("manufacturer", ""))
        pkey = str(r["product_key"])
        pname = str(r["product_name"])
        doc = str(r.get("embedding_text", ""))

        meta = {
            "batch_date": batch_date,
            "mall_name": mall,
            "manufacturer": manu,
            "product_key": pkey,
            "product_name": pname,
            "chunk_type": "daily_summary",
            "min_price": int(r["min_price"]) if pd.notna(r.get("min_price")) else None,
            "avg_price": float(r["avg_price"]) if pd.notna(r.get("avg_price")) else None,
            "max_price": int(r["max_price"]) if pd.notna(r.get("max_price")) else None,
            "source": "excel",
            "version": version,
        }

        ids.append(make_id(batch_date, mall, pkey, "daily_summary"))
        docs.append(doc)
        metas.append(meta)

        if len(ids) >= batch_size:
            col_sum.upsert(ids=ids, documents=docs, metadatas=metas)
            ids, docs, metas = [], [], []

    if ids:
        col_sum.upsert(ids=ids, documents=docs, metadatas=metas)


def upsert_reviews_digest(col_rev, df_data: pd.DataFrame, batch_size: int = 500, version: str = "redesigned_v1"):
    key_cols = ["batch_date", "mall_name", "manufacturer", "product_key", "product_name"]
    grouped = df_data.groupby(key_cols, dropna=False)

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    for (batch_date, mall, manu, pkey, pname), g in grouped:
        batch_date = str(batch_date)
        mall = str(mall)
        manu = str(manu)
        pkey = str(pkey)
        pname = str(pname)

        top = g[g["rank"].isin([1, 2, 3, 4, 5])] if "rank" in g.columns else g
        if len(top) == 0:
            top = g

        if "comments_top5" in top.columns:
            raw = " / ".join(top["comments_top5"].astype(str).tolist())
        else:
            raw = " / ".join(top.astype(str).agg(" | ".join, axis=1).tolist())

        raw = raw[:2000]
        doc = (
            f"[댓글요약] 날짜={batch_date}\n"
            f"제조사={manu}\n"
            f"제품={pname}({pkey})\n"
            f"쇼핑몰={mall}\n\n"
            f"원문:\n{raw}"
        )

        meta = {
            "batch_date": batch_date,
            "mall_name": mall,
            "manufacturer": manu,
            "product_key": pkey,
            "product_name": pname,
            "chunk_type": "reviews_digest",
            "source": "excel",
            "version": version,
        }

        ids.append(make_id(batch_date, mall, pkey, "reviews_digest"))
        docs.append(doc)
        metas.append(meta)

        if len(ids) >= batch_size:
            col_rev.upsert(ids=ids, documents=docs, metadatas=metas)
            ids, docs, metas = [], [], []

    if ids:
        col_rev.upsert(ids=ids, documents=docs, metadatas=metas)


# ============================================================
# LLM helpers
# ============================================================
def _remove_chat_tokens(text: str) -> str:
    t = text.replace("\r\n", "\n")
    t = re.sub(r"<\|.*?\|>", "", t)
    return t.strip()


def _extract_json_object(text: str) -> str:
    t = _remove_chat_tokens(text)
    m = re.search(r"```(?:json)?\s*(.*?)```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        t = m.group(1).strip()
    start = t.find("{")
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1].strip()
    return ""


def init_local_llm(model_path: str, n_ctx: int, n_threads: int, n_gpu_layers: int):
    global LLM
    if LLM is not None:
        return LLM

    if int(n_gpu_layers) == 0 and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        from llama_cpp import Llama  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError(f"llama-cpp-python import 실패: {repr(e)}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")

    LLM = Llama(
        model_path=model_path,
        n_ctx=int(n_ctx),
        n_threads=int(n_threads),
        n_gpu_layers=int(n_gpu_layers),
        verbose=False,
    )
    return LLM


def build_llm_intent_context(today: str, default_start: str) -> str:
    schema_line = (
        '{'
        '"intent": "Q1" | "Q1_DETAIL" | "Q4" | "Q5" | "Q5_RANGE_SUMMARY" | "Q5_RANGE_DETAIL" | "Q6" | "Q7" | "Q8_TREND" | '
        '"Q7B_MANU_RANGE_TOPN" | "Q7C_MANU_RANGE_TOPN_WITH_DATE" | "Q7D_MANU_RANGE_TOPN_BY_MALL" | '
        '"Q9_MALL_SUMMARY_TABLE" | "A1_VIOL_TREND" | "A1_VIOL_DETAIL" | "UNKNOWN", '
        '"signals": {"has_violation": true|false, "has_detail": true|false, "has_trend": true|false}'
        '}'
    )
    return (
        "You are a STRICT classifier. Convert the Korean user question into ONE JSON object.\n"
        "Output ONLY one JSON object. No explanation. No markdown.\n\n"
        f"today(batch_date)={today}, default_start={default_start}\n\n"
        "Return schema:\n"
        f"{schema_line}\n\n"
    )


def validate_llm_intent(intent: str, *, mall: Optional[str], pkey: Optional[str], manu: Optional[str]) -> bool:
    if intent in ["A1_VIOL_DETAIL", "A1_VIOL_TREND", "Q5", "Q5_RANGE_SUMMARY", "Q5_RANGE_DETAIL", "Q6", "Q7", "Q9_MALL_SUMMARY_TABLE"] and not mall:
        return False
    if intent in ["Q1", "Q1_DETAIL", "Q8_TREND"] and not pkey:
        return False
    if intent in ["Q7B_MANU_RANGE_TOPN", "Q7C_MANU_RANGE_TOPN_WITH_DATE", "Q7D_MANU_RANGE_TOPN_BY_MALL"] and not manu:
        return False
    return True


def llm_classify_intent(
    question: str,
    *,
    today: str,
    default_start: str,
    model_path: str,
    n_ctx: int,
    max_tokens: int,
    temperature: float,
    n_threads: int,
    n_gpu_layers: int,
    mall: Optional[str],
    pkey: Optional[str],
    manu: Optional[str],
) -> str:
    ctx = build_llm_intent_context(today=today, default_start=default_start)
    llm = init_local_llm(model_path, n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=n_gpu_layers)
    prompt = f"{ctx}\n\nUser question (Korean): {question}\n\nJSON:\n"
    out = llm(prompt, max_tokens=int(max_tokens), temperature=float(temperature), stop=["```", "<|end|>", "<|start|>"])
    raw = out["choices"][0]["text"]
    jtxt = _extract_json_object(raw)
    if not jtxt:
        return "UNKNOWN"
    try:
        obj = json.loads(jtxt)
    except Exception:
        return "UNKNOWN"

    intent = str(obj.get("intent") or "").strip()
    allow = {
        "Q1", "Q1_DETAIL", "Q4", "Q5", "Q5_RANGE_SUMMARY", "Q5_RANGE_DETAIL", "Q6", "Q7", "Q8_TREND",
        "Q7B_MANU_RANGE_TOPN", "Q7C_MANU_RANGE_TOPN_WITH_DATE", "Q7D_MANU_RANGE_TOPN_BY_MALL",
        "Q9_MALL_SUMMARY_TABLE", "A1_VIOL_TREND", "A1_VIOL_DETAIL", "UNKNOWN"
    }
    if intent not in allow:
        return "UNKNOWN"

    if not validate_llm_intent(intent, mall=mall, pkey=pkey, manu=manu):
        return "UNKNOWN"
    return intent


# ============================================================
# Plan fallback
# ============================================================
def print_normalized_plan(plan: PandasPlan):
    plan_dict = {
        "source": plan.source,
        "select": plan.select,
        "filters": [asdict(x) for x in plan.filters],
        "groupby": plan.groupby,
        "aggregations": [asdict(x) for x in plan.aggregations],
        "sort": [asdict(x) for x in plan.sort],
        "limit": plan.limit,
    }
    print("\n[NORMALIZED PLAN]")
    print(json.dumps(plan_dict, ensure_ascii=False, indent=2))


def infer_plan_source_from_question(q: str) -> str:
    if prefers_summary_semantic(q):
        return "df_pc_all"
    if has_generic_price_bucket_words(q):
        return "df_pc_all"
    if prefers_raw_semantic(q):
        return "df_data_all"
    if any(k in q for k in ["최저가", "최고가", "평균가", "평균", "가격", "시세", "상위", "하위", "순위", "추이", "기간", "최근", "지난", "싼", "비싼", "저렴"]):
        return "df_pc_all"
    return "df_data_all"


def build_llm_plan_context(
    df_pc_all: pd.DataFrame,
    df_data_all: pd.DataFrame,
    today: str,
    question: str,
    entities: Dict[str, Any],
    slots: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> str:
    def brief_cols(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        cols_show = cols[:90]
        return ", ".join(cols_show) + ("" if len(cols_show) == len(cols) else f" ...(+{len(cols)-len(cols_show)})")

    return (
        "You convert a Korean user question into ONE STRICT JSON PandasPlan.\n"
        "Output ONLY one JSON object. No explanation. No markdown. No prose.\n\n"
        f"today(batch_date)={today}\n"
        f"question={question}\n"
        f"resolved_entities={json.dumps(entities, ensure_ascii=False)}\n"
        f"semantic_slots={json.dumps(slots, ensure_ascii=False)}\n"
        f"resolved_date_range={{\"start_date\":\"{start_date}\",\"end_date\":\"{end_date}\"}}\n\n"
        "Required top-level schema:\n"
        "{\n"
        '  "source": "df_pc_all" | "df_data_all",\n'
        '  "select": ["col1","col2"],\n'
        '  "filters": [{"col":"batch_date","op":"=","value":"YYYY-MM-DD"}],\n'
        '  "groupby": ["colA"],\n'
        '  "aggregations": [{"col":"avg_price","func":"mean","as":"avg_price"}],\n'
        '  "sort": [{"col":"avg_price","dir":"desc"}],\n'
        '  "limit": 20\n'
        "}\n\n"
        "Rules:\n"
        "- ALWAYS include all top-level keys.\n"
        "- Use only existing columns from source.\n"
        "- Support filter ops: =, !=, contains, between, in, not in.\n"
        "- Generic cheap/expensive questions should prefer df_pc_all summary.\n"
        "- If prefers summary/price table/min/avg/max/topN/trend -> df_pc_all.\n"
        "- If raw/original/offers/detail -> df_data_all.\n"
        "- If entities.mall exists, add mall_name filter.\n"
        "- If entities.mall_list has multiple values, use op='in'.\n"
        "- If entities.manu exists, add manufacturer filter.\n"
        "- If entities.manu_list has multiple values, use op='in'.\n"
        "- If entities.pkey exists, add product_key '=' filter.\n"
        "- Use resolved_date_range as batch_date filter.\n"
        "- For df_data_all raw output, prefer sort by rank asc or price asc.\n"
        "- For detail/raw output, include rank, price, item_name, url, comments_top5, date when available.\n"
        "- For generic range questions without explicit entity, include batch_date in select.\n"
        "- If comparing manufacturers only, aggregate by manufacturer.\n"
        "- If comparing malls only, aggregate by mall_name.\n"
        "- Do not attempt violation/not-cheapest business logic.\n\n"
        f"df_pc_all columns: {brief_cols(df_pc_all)}\n"
        f"df_data_all columns: {brief_cols(df_data_all)}\n"
    )


def llm_generate_plan(
    question: str,
    ctx: str,
    model_path: str,
    n_ctx: int,
    max_tokens: int,
    temperature: float,
    n_threads: int,
    n_gpu_layers: int,
) -> Dict[str, Any]:
    llm = init_local_llm(model_path, n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=n_gpu_layers)

    prompts = [
        f"{ctx}\n\nJSON:\n",
        (
            "Return ONLY one valid JSON object.\n"
            "No explanation.\n"
            "No prose.\n"
            "Start with '{' and end with '}'.\n\n"
            f"{ctx}\n\nJSON:\n"
        ),
    ]

    last_raw = ""
    for prompt in prompts:
        try:
            out = llm(prompt, max_tokens=int(max_tokens), temperature=float(temperature), stop=["```", "<|end|>", "<|start|>"])
            raw = out["choices"][0]["text"]
            last_raw = raw
            jtxt = _extract_json_object(raw)
            if not jtxt:
                continue
            try:
                return json.loads(jtxt)
            except Exception:
                continue
        except Exception as e:
            last_raw = f"[LLM_CALL_ERROR] {repr(e)}"
            continue

    raise ValueError(f"LLM output did not contain valid JSON object. raw={last_raw[:300]!r}")


def _normalize_list_or_empty(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return []


def _append_filter_if_missing(
    filters: List[Dict[str, Any]],
    col: str,
    op: str,
    value: Any = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    for f in filters:
        if not isinstance(f, dict):
            continue
        if str(f.get("col") or "") == col:
            return
    if op == "between":
        filters.append({"col": col, "op": "between", "start": start, "end": end})
    else:
        filters.append({"col": col, "op": op, "value": value})


def build_rule_based_miniplan(
    q: str,
    *,
    batch_date: str,
    start_date: str,
    end_date: str,
    mall: Optional[str],
    manu: Optional[str],
    pkey: Optional[str],
    mall_list: List[str],
    manu_list: List[str],
) -> Dict[str, Any]:
    metric = parse_metric_kor(q)
    ascending = parse_sort_direction(q, metric)
    sort_dir = "asc" if ascending else "desc"
    source = infer_plan_source_from_question(q)

    filters: List[Dict[str, Any]] = []
    if start_date != end_date:
        filters.append({"col": "batch_date", "op": "between", "start": start_date, "end": end_date})
    else:
        filters.append({"col": "batch_date", "op": "=", "value": batch_date})

    if len(mall_list) >= 2:
        filters.append({"col": "mall_name", "op": "in", "value": mall_list})
    elif mall:
        filters.append({"col": "mall_name", "op": "=", "value": mall})

    if len(manu_list) >= 2:
        filters.append({"col": "manufacturer", "op": "in", "value": manu_list})
    elif manu:
        filters.append({"col": "manufacturer", "op": "=", "value": manu})

    if pkey:
        filters.append({"col": "product_key", "op": "=", "value": pkey})

    if source == "df_pc_all":
        limit = summary_limit_for_question(q)

        if has_compare_words(q) and len(manu_list) >= 2 and len(mall_list) == 0 and not pkey:
            return {
                "source": "df_pc_all",
                "select": ["manufacturer", "value"],
                "filters": filters,
                "groupby": ["manufacturer"],
                "aggregations": [{"col": metric, "func": "mean", "as": "value"}],
                "sort": [{"col": "value", "dir": sort_dir}],
                "limit": max(2, len(manu_list)),
            }

        if has_compare_words(q) and len(mall_list) >= 2 and not pkey:
            group_cols = ["mall_name"]
            if manu:
                group_cols = ["manufacturer", "mall_name"]
            return {
                "source": "df_pc_all",
                "select": group_cols + ["value"],
                "filters": filters,
                "groupby": group_cols,
                "aggregations": [{"col": metric, "func": "mean", "as": "value"}],
                "sort": [{"col": "value", "dir": sort_dir}],
                "limit": max(2, len(mall_list) * max(1, len(manu_list))),
            }

        if start_date != end_date and not mall and not manu and not pkey:
            return {
                "source": "df_pc_all",
                "select": ["batch_date", "manufacturer", "mall_name", "product_key", "product_name", metric],
                "filters": filters,
                "groupby": [],
                "aggregations": [],
                "sort": [{"col": "batch_date", "dir": "asc"}, {"col": metric, "dir": sort_dir}],
                "limit": limit,
            }

        if wants_date_in_rows(q):
            return {
                "source": "df_pc_all",
                "select": ["batch_date", "manufacturer", "mall_name", "product_key", "product_name", metric],
                "filters": filters,
                "groupby": [],
                "aggregations": [],
                "sort": [{"col": "batch_date", "dir": "asc"}, {"col": metric, "dir": sort_dir}],
                "limit": limit,
            }

        return {
            "source": "df_pc_all",
            "select": ["manufacturer", "mall_name", "product_key", "product_name", metric],
            "filters": filters,
            "groupby": [],
            "aggregations": [],
            "sort": [{"col": metric, "dir": sort_dir}],
            "limit": limit,
        }

    limit = raw_limit_for_question(q)
    return {
        "source": "df_data_all",
        "select": DETAIL_RAW_DEFAULT_SELECT,
        "filters": filters,
        "groupby": [],
        "aggregations": [],
        "sort": [{"col": "rank", "dir": "asc"}],
        "limit": limit,
    }


def repair_plan_structure(
    raw_plan: Dict[str, Any],
    *,
    question: str,
    batch_date: str,
    start_date: str,
    end_date: str,
    mall: Optional[str],
    manu: Optional[str],
    pkey: Optional[str],
    mall_list: List[str],
    manu_list: List[str],
) -> Dict[str, Any]:
    q = question.strip()
    plan = raw_plan if isinstance(raw_plan, dict) else {}

    source = plan.get("source")
    if source not in ("df_pc_all", "df_data_all"):
        source = infer_plan_source_from_question(q)

    select = _normalize_list_or_empty(plan.get("select"))
    filters = _normalize_list_or_empty(plan.get("filters"))
    groupby = _normalize_list_or_empty(plan.get("groupby"))
    aggregations = _normalize_list_or_empty(plan.get("aggregations"))
    sort = _normalize_list_or_empty(plan.get("sort"))

    limit = plan.get("limit")
    try:
        limit = int(limit)
    except Exception:
        limit = raw_limit_for_question(q) if source == "df_data_all" else summary_limit_for_question(q)

    repaired = {
        "source": source,
        "select": select,
        "filters": filters,
        "groupby": groupby,
        "aggregations": aggregations,
        "sort": sort,
        "limit": max(1, min(HARD_LIMIT_ROWS, int(limit))),
    }

    if start_date != end_date:
        _append_filter_if_missing(repaired["filters"], "batch_date", "between", start=start_date, end=end_date)
    else:
        _append_filter_if_missing(repaired["filters"], "batch_date", "=", value=batch_date)

    if len(mall_list) >= 2:
        _append_filter_if_missing(repaired["filters"], "mall_name", "in", value=mall_list)
    elif mall:
        _append_filter_if_missing(repaired["filters"], "mall_name", "=", value=mall)

    if len(manu_list) >= 2:
        _append_filter_if_missing(repaired["filters"], "manufacturer", "in", value=manu_list)
    elif manu:
        _append_filter_if_missing(repaired["filters"], "manufacturer", "=", value=manu)

    if pkey:
        _append_filter_if_missing(repaired["filters"], "product_key", "=", value=pkey)

    if not repaired["select"]:
        if repaired["source"] == "df_pc_all":
            repaired["select"] = SUMMARY_DEFAULT_SELECT
        else:
            repaired["select"] = DETAIL_RAW_DEFAULT_SELECT

    return repaired


def enrich_plan_semantics(
    plan: Dict[str, Any],
    *,
    question: str,
    start_date: str,
    end_date: str,
    mall: Optional[str],
    manu: Optional[str],
    pkey: Optional[str],
    mall_list: List[str],
    manu_list: List[str],
) -> Dict[str, Any]:
    q = question.strip()
    metric = parse_metric_kor(q)

    if prefers_summary_semantic(q) or has_generic_price_bucket_words(q):
        plan["source"] = "df_pc_all"
    if prefers_raw_semantic(q) and not prefers_summary_semantic(q):
        plan["source"] = "df_data_all"

    if plan["source"] == "df_data_all":
        plan["limit"] = raw_limit_for_question(q)
        plan["select"] = [c for c in DETAIL_RAW_DEFAULT_SELECT if c]
        if not plan.get("sort"):
            plan["sort"] = [{"col": "rank", "dir": "asc"}]

    if plan["source"] == "df_pc_all":
        plan["limit"] = min(HARD_LIMIT_ROWS, int(plan.get("limit", summary_limit_for_question(q))))
        if not pkey and not plan.get("groupby"):
            if wants_date_in_rows(q) or (start_date != end_date and not mall and not manu):
                plan["select"] = ["batch_date", "manufacturer", "mall_name", "product_key", "product_name", metric]
            else:
                plan["select"] = ["manufacturer", "mall_name", "product_key", "product_name", metric]
        if not plan.get("sort"):
            plan["sort"] = [{"col": metric, "dir": "asc" if parse_sort_direction(q, metric) else "desc"}]

    if wants_date_in_rows(q) and plan["source"] == "df_pc_all":
        if "batch_date" not in plan["select"]:
            plan["select"] = ["batch_date"] + [c for c in plan["select"] if c != "batch_date"]
        plan["groupby"] = []
        plan["aggregations"] = []
        plan["sort"] = [
            {"col": "batch_date", "dir": "asc"},
            {"col": metric, "dir": "asc" if parse_sort_direction(q, metric) else "desc"},
        ]

    if plan["source"] == "df_pc_all" and start_date != end_date and not mall and not manu and not pkey and not has_compare_words(q):
        plan["groupby"] = []
        plan["aggregations"] = []
        plan["select"] = ["batch_date", "manufacturer", "mall_name", "product_key", "product_name", metric]
        plan["sort"] = [
            {"col": "batch_date", "dir": "asc"},
            {"col": metric, "dir": "asc" if parse_sort_direction(q, metric) else "desc"},
        ]

    if (
        plan["source"] == "df_pc_all"
        and manu
        and start_date != end_date
        and not pkey
        and not wants_date_in_rows(q)
        and not wants_group_by_mall(q)
        and not has_compare_words(q)
        and any(k in q for k in ["평균가", "평균", "최저가", "최고가", "상위", "하위", "높은", "낮은", "싼", "비싼"])
    ):
        agg_as = "value"
        groupby_cols = ["manufacturer", "mall_name", "product_key", "product_name"]
        plan["groupby"] = groupby_cols
        plan["aggregations"] = [{"col": metric, "func": "mean", "as": agg_as}]
        plan["select"] = groupby_cols + [agg_as]
        plan["sort"] = [{"col": agg_as, "dir": "asc" if parse_sort_direction(q, metric) else "desc"}]
        plan["limit"] = summary_limit_for_question(q)

    if has_compare_words(q) and plan["source"] == "df_pc_all" and len(manu_list) >= 2 and len(mall_list) == 0 and not pkey:
        plan["groupby"] = ["manufacturer"]
        plan["aggregations"] = [{"col": metric, "func": "mean", "as": "value"}]
        plan["select"] = ["manufacturer", "value"]
        plan["sort"] = [{"col": "value", "dir": "asc" if parse_sort_direction(q, metric) else "desc"}]
        plan["limit"] = max(2, len(manu_list))

    if has_compare_words(q) and plan["source"] == "df_pc_all" and len(mall_list) >= 2 and not pkey:
        group_cols = ["mall_name"]
        if manu:
            group_cols = ["manufacturer", "mall_name"]
        plan["groupby"] = group_cols
        plan["aggregations"] = [{"col": metric, "func": "mean", "as": "value"}]
        plan["select"] = group_cols + ["value"]
        plan["sort"] = [{"col": "value", "dir": "asc" if parse_sort_direction(q, metric) else "desc"}]
        plan["limit"] = max(2, len(mall_list) * max(1, len(manu_list)))

    return plan


def validate_plan_against_question(
    q: str,
    plan: Dict[str, Any],
    *,
    mall: Optional[str],
    manu: Optional[str],
    pkey: Optional[str],
    mall_list: List[str],
    manu_list: List[str],
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    filters = _normalize_list_or_empty(plan.get("filters"))

    def has_filter(col: str) -> bool:
        for f in filters:
            if isinstance(f, dict) and str(f.get("col") or "") == col:
                return True
        return False

    if len(mall_list) >= 2 and not has_filter("mall_name"):
        _append_filter_if_missing(filters, "mall_name", "in", value=mall_list)
    elif mall and not has_filter("mall_name"):
        _append_filter_if_missing(filters, "mall_name", "=", value=mall)

    if len(manu_list) >= 2 and not has_filter("manufacturer"):
        _append_filter_if_missing(filters, "manufacturer", "in", value=manu_list)
    elif manu and not has_filter("manufacturer"):
        _append_filter_if_missing(filters, "manufacturer", "=", value=manu)

    if pkey and not has_filter("product_key"):
        _append_filter_if_missing(filters, "product_key", "=", value=pkey)

    if not has_filter("batch_date"):
        if start_date != end_date:
            _append_filter_if_missing(filters, "batch_date", "between", start=start_date, end=end_date)
        else:
            _append_filter_if_missing(filters, "batch_date", "=", value=start_date)

    plan["filters"] = filters

    if has_generic_price_bucket_words(q):
        plan["source"] = "df_pc_all"
        metric = parse_metric_kor(q)
        if start_date != end_date and not mall and not manu and not pkey:
            plan["select"] = ["batch_date", "manufacturer", "mall_name", "product_key", "product_name", metric]
            plan["sort"] = [{"col": "batch_date", "dir": "asc"}, {"col": metric, "dir": "asc" if parse_sort_direction(q, metric) else "desc"}]
        else:
            plan["select"] = ["manufacturer", "mall_name", "product_key", "product_name", metric]
            plan["sort"] = [{"col": metric, "dir": "asc" if parse_sort_direction(q, metric) else "desc"}]
        plan["groupby"] = []
        plan["aggregations"] = []
        plan["limit"] = summary_limit_for_question(q)

    if prefers_summary_semantic(q):
        plan["source"] = "df_pc_all"
    if prefers_raw_semantic(q) and not prefers_summary_semantic(q):
        plan["source"] = "df_data_all"

    if plan["source"] == "df_data_all":
        plan["limit"] = raw_limit_for_question(q)
        plan["select"] = DETAIL_RAW_DEFAULT_SELECT
        sort_cols = [s.get("col") for s in _normalize_list_or_empty(plan.get("sort")) if isinstance(s, dict)]
        if not sort_cols or "min_price" in sort_cols or "avg_price" in sort_cols or "max_price" in sort_cols:
            plan["sort"] = [{"col": "rank", "dir": "asc"}]

    return plan


def _clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(x)
    except Exception:
        return default
    return max(lo, min(hi, v))


def validate_and_normalize_plan(plan: Dict[str, Any], df_pc_all: pd.DataFrame, df_data_all: pd.DataFrame) -> PandasPlan:
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object.")

    source = plan.get("source")
    if source not in ("df_pc_all", "df_data_all"):
        raise ValueError(f"Invalid source: {source}")

    df = df_pc_all if source == "df_pc_all" else df_data_all
    cols_set = set(df.columns)

    select = plan.get("select") or []
    if not isinstance(select, list):
        raise ValueError("select must be list")
    select2 = [str(c) for c in select if str(c) in cols_set]
    if not select2:
        base = ["batch_date", "mall_name", "manufacturer", "product_key", "product_name"]
        select2 = [c for c in base if c in cols_set]

    raw_filters = plan.get("filters") or []
    if not isinstance(raw_filters, list):
        raise ValueError("filters must be list")
    filters2: List[PandasFilter] = []
    for f in raw_filters[:80]:
        if not isinstance(f, dict):
            continue
        col = str(f.get("col") or "")
        if col not in cols_set:
            continue
        op = f.get("op")
        if op not in ("=", "!=", "contains", "between", "in", "not in"):
            continue
        if op == "between":
            start = f.get("start")
            end = f.get("end")
            if not start or not end:
                continue
            filters2.append(PandasFilter(col=col, op="between", start=str(start)[:10], end=str(end)[:10]))
        else:
            val = f.get("value")
            if op in ("in", "not in"):
                if not isinstance(val, list) or len(val) == 0:
                    continue
                filters2.append(PandasFilter(col=col, op=op, value=val))
            elif op == "contains":
                if val is None:
                    continue
                filters2.append(PandasFilter(col=col, op="contains", value=str(val)))
            else:
                if val is None:
                    continue
                filters2.append(PandasFilter(col=col, op=op, value=val))

    groupby = plan.get("groupby") or []
    if not isinstance(groupby, list):
        raise ValueError("groupby must be list")
    groupby2 = [str(c) for c in groupby if str(c) in cols_set]

    aggregations = plan.get("aggregations") or []
    if not isinstance(aggregations, list):
        raise ValueError("aggregations must be list")
    aggs2: List[PandasAgg] = []
    for a in aggregations[:40]:
        if not isinstance(a, dict):
            continue
        col = str(a.get("col") or "")
        func = a.get("func")
        as_name = str(a.get("as") or "").strip()
        if col not in cols_set:
            continue
        if func not in ("count", "sum", "mean", "min", "max", "nunique"):
            continue
        if not as_name:
            as_name = f"{func}_{col}"
        aggs2.append(PandasAgg(col=col, func=func, as_name=as_name[:64]))

    raw_sort = plan.get("sort") or []
    if not isinstance(raw_sort, list):
        raise ValueError("sort must be list")
    sort2: List[PandasSort] = []
    agg_aliases = {a.as_name for a in aggs2}
    for s in raw_sort[:10]:
        if not isinstance(s, dict):
            continue
        col = str(s.get("col") or "")
        direction = s.get("dir")
        if direction not in ("asc", "desc"):
            continue
        if col not in cols_set and col not in agg_aliases:
            continue
        sort2.append(PandasSort(col=col, dir=direction))

    default_limit = DEFAULT_RAW_LIMIT if source == "df_data_all" else DEFAULT_SUMMARY_LIMIT
    limit = _clamp_int(plan.get("limit"), 1, HARD_LIMIT_ROWS, default_limit)

    return PandasPlan(
        source=source,
        select=select2[:50],
        filters=filters2,
        groupby=groupby2[:10],
        aggregations=aggs2,
        sort=sort2,
        limit=limit,
    )


def execute_plan(plan: PandasPlan, df_pc_all: pd.DataFrame, df_data_all: pd.DataFrame) -> pd.DataFrame:
    df = df_pc_all if plan.source == "df_pc_all" else df_data_all
    cur = df

    for f in plan.filters:
        if f.op == "between":
            cur = cur[(cur[f.col] >= str(f.start)) & (cur[f.col] <= str(f.end))]
        elif f.op == "contains":
            cur = cur[cur[f.col].astype(str).str.contains(str(f.value), na=False)]
        elif f.op == "=":
            cur = cur[cur[f.col] == f.value]
        elif f.op == "!=":
            cur = cur[cur[f.col] != f.value]
        elif f.op == "in":
            vals = f.value if isinstance(f.value, list) else [f.value]
            cur = cur[cur[f.col].isin(vals)]
        elif f.op == "not in":
            vals = f.value if isinstance(f.value, list) else [f.value]
            cur = cur[~cur[f.col].isin(vals)]

    cur = cur.copy()

    if plan.groupby and plan.aggregations:
        agg_dict: Dict[str, List[str]] = {}
        alias_map: Dict[Tuple[str, str], str] = {}
        for a in plan.aggregations:
            agg_dict.setdefault(a.col, []).append(a.func)
            alias_map[(a.col, a.func)] = a.as_name

        grp = cur.groupby(plan.groupby, dropna=False).agg(agg_dict)
        new_cols: List[str] = []
        for c0, c1 in grp.columns.to_list():
            new_cols.append(alias_map.get((c0, c1), f"{c0}_{c1}"))
        grp.columns = new_cols
        out = grp.reset_index()
    else:
        out = cur

    if plan.select:
        keep = [c for c in plan.select if c in out.columns]
        agg_alias_cols = [a.as_name for a in plan.aggregations if a.as_name in out.columns]
        for c in agg_alias_cols:
            if c not in keep:
                keep.append(c)
        if keep:
            out = out[keep].copy()

    if plan.sort:
        sort_cols = []
        ascending = []
        for s in plan.sort:
            if s.col in out.columns:
                sort_cols.append(s.col)
                ascending.append(s.dir == "asc")
        if sort_cols:
            out = out.sort_values(sort_cols, ascending=ascending)

    out = out.head(int(plan.limit)).copy()
    return out


def should_block_plan_fallback(q: str, slots: SemanticSlots) -> bool:
    if slots.has_compare_words:
        return False
    return any(k in q for k in PLAN_FORBIDDEN_PATTERNS)


# ============================================================
# Query resolution
# ============================================================
def resolve_entities(q: str, catalog: Dict[str, Any]) -> ParsedEntities:
    pkey = parse_product_key(q, catalog)
    mall = extract_target_mall(q, catalog["malls"])
    manu = parse_manufacturer(q, catalog["manufacturers"])
    mall_list = extract_all_malls(q, catalog["malls"])
    manu_list = parse_all_manufacturers(q, catalog["manufacturers"])

    if mall is None:
        mall = _fuzzy_pick(q, catalog["malls"], cutoff=FUZZY_CUTOFF)
    if manu is None:
        manu = _fuzzy_pick(q, catalog["manufacturers"], cutoff=FUZZY_CUTOFF)

    if mall and mall not in mall_list:
        mall_list.append(mall)
    if manu and manu not in manu_list:
        manu_list.append(manu)

    my_manu = resolve_my_manufacturer(q, catalog["manufacturers"], fallback_my=MY_MANUFACTURER)
    if not manu and my_manu:
        manu = my_manu
    if my_manu and my_manu not in manu_list:
        manu_list.append(my_manu)

    return ParsedEntities(
        product_key=pkey,
        mall=mall,
        manufacturer=manu,
        mall_list=mall_list,
        manufacturer_list=manu_list,
    )


def build_query_context(app: AppContext, raw_question: str) -> QueryContext:
    q = normalize_range_separators(raw_question.strip())
    entities = resolve_entities(q, app.catalog)
    dates = resolve_dates(q, default_end=app.today, default_start=app.default_start)
    slots = extract_semantic_slots(q)
    intent = detect_intent(entities, dates, slots)

    return QueryContext(
        raw_question=raw_question,
        question=q,
        entities=entities,
        dates=dates,
        slots=slots,
        intent=intent,
    )


# ============================================================
# Renderer / notices
# ============================================================
def print_date_clamp_notice(qc: QueryContext, today: str):
    d = qc.dates
    if d.clamped_end and d.requested_end > today:
        print(f"[안내] 종료일 {d.requested_end}은 데이터 범위를 벗어나 {today}까지로 계산했습니다.")
    if d.clamped_start and d.requested_start < d.resolved_start:
        print(f"[안내] 시작일 {d.requested_start}은 데이터 시작 범위를 벗어나 {d.resolved_start}부터로 계산했습니다.")


def print_parsed_debug(qc: QueryContext, plan_allowed: bool):
    entities_for_debug = {
        "pkey": {"value": qc.entities.product_key},
        "mall": {"value": qc.entities.mall},
        "manu": {"value": qc.entities.manufacturer},
        "mall_list": qc.entities.mall_list,
        "manu_list": qc.entities.manufacturer_list,
        "requested_start": qc.dates.requested_start,
        "requested_end": qc.dates.requested_end,
        "start_date": qc.dates.resolved_start,
        "end_date": qc.dates.resolved_end,
        "batch_date": qc.dates.batch_date,
        "date_source": qc.dates.source,
        "is_range": qc.dates.is_range,
    }
    print_debug_json({
        "event": "parsed_question",
        "q": qc.question,
        "entities": entities_for_debug,
        "slots": asdict(qc.slots),
        "intent_before_llm": qc.intent,
        "plan_allowed": plan_allowed,
    })


# ============================================================
# Structured executor
# ============================================================
def execute_structured_route(app: AppContext, qc: QueryContext) -> bool:
    args = app.args
    q = qc.question
    e = qc.entities
    d = qc.dates
    intent = qc.intent

    if intent in ["A1_VIOL_TREND", "A1_VIOL_DETAIL"] and not e.manufacturer:
        e.manufacturer = MY_MANUFACTURER

    if intent == "A1_VIOL_TREND":
        if not e.mall or not e.manufacturer:
            print("쇼핑몰 또는 제조사를 인식 못했어요.\n")
            return True
        print_date_clamp_notice(qc, app.today)
        out_df = a1_violation_trend(app.df_pc_all, d.resolved_start, d.resolved_end, e.manufacturer, e.mall)
        if out_df is None or len(out_df) == 0:
            print("기간 내 데이터가 없습니다.\n")
            return True
        title_range = d.resolved_start if not d.is_range else f"{d.resolved_start} ~ {d.resolved_end}"
        print(f"\n[A급 정형] {e.manufacturer} '{e.mall}' 위반 추이 ({title_range})")
        print_result_any(out_df, output_dir=args.output_dir, prefix=f"A1_VIOL_TREND_{e.manufacturer}_{e.mall}_{d.resolved_start}_to_{d.resolved_end}")
        print()
        return True

    if intent == "A1_VIOL_DETAIL":
        if not e.mall or not e.manufacturer:
            print("쇼핑몰 또는 제조사를 인식 못했어요.\n")
            return True
        print_date_clamp_notice(qc, app.today)
        out_df = a1_violation_detail(app.df_pc_all, d.resolved_start, d.resolved_end, e.manufacturer, e.mall, limit=5000)
        if out_df is None:
            print("기간 내 데이터가 없습니다.\n")
            return True
        title_range = d.resolved_start if not d.is_range else f"{d.resolved_start} ~ {d.resolved_end}"
        if len(out_df) == 0:
            print(f"\n[A급 정형] {e.manufacturer} '{e.mall}' 위반 상세 ({title_range})")
            print("(위반 없음)\n")
            return True
        product_count = int(out_df["product_key"].nunique()) if "product_key" in out_df.columns else 0
        case_count = int(len(out_df))
        print(f"\n[A급 정형] {e.manufacturer} '{e.mall}' 위반 상세 ({title_range}) — 제품 {product_count}개 / 케이스 {case_count}행")
        print_result_any(out_df, output_dir=args.output_dir, prefix=f"A1_VIOL_DETAIL_{e.manufacturer}_{e.mall}_{d.resolved_start}_to_{d.resolved_end}")
        print()
        return True

    if intent == "Q1":
        if not e.product_key:
            print("제품을 인식 못했어요. (엑셀에 존재하는 제품코드/제품명이 질문에 포함되어야 함)\n")
            return True

        run_dates = expand_date_range_days(d.resolved_start, d.resolved_end, max_days=45) if d.is_range else [d.batch_date]

        any_found = False
        for one_date in run_dates:
            out = q1_product_best(app.df_pc_all, e.product_key, one_date)
            if not out:
                print(f"\n[정형결과] {e.product_key} {one_date} 데이터가 없습니다.")
                continue
            any_found = True
            manu_txt = out.get("manufacturer") or app.catalog["key_to_manufacturer"].get(e.product_key, "-")
            print(f"\n[정형결과] {e.product_key} {one_date} 제조사={manu_txt} 최저가: {out['best_mall']} / {out['best_price']}원")
            print(df_to_string_kr(out["table"], index=False))

        print()
        if not any_found:
            print("데이터를 못 찾았어요.\n")
        return True

    if intent == "Q1_DETAIL":
        if not e.product_key:
            print("제품을 인식 못했어요. (엑셀에 존재하는 제품코드/제품명이 질문에 포함되어야 함)\n")
            return True
        run_date = d.batch_date
        offers = q1_product_all_offers(app.df_data_all, e.product_key, run_date)
        if offers is None:
            print(f"\n[정형결과] {e.product_key} {run_date} 오퍼(원본) 데이터를 못 찾았어요.\n")
            return True
        print(f"\n[정형결과] {e.product_key} {run_date} 오퍼 목록 {len(offers)}건")
        print_result_any(offers, output_dir=args.output_dir, prefix=f"Q1_DETAIL_{e.product_key}_{run_date}")
        print()
        return True

    if intent == "Q4":
        out = q4_shipping_issues(app.df_data_all, d.resolved_start, d.resolved_end, product_key=e.product_key if e.product_key else None)
        if out is None or len(out) == 0:
            print("데이터를 못 찾았어요.\n")
            return True
        n = parse_top_n(q, default_n=10)
        title_range = d.resolved_start if not d.is_range else f"{d.resolved_start}~{d.resolved_end}"
        print(f"\n[정형결과] 배송 이슈 많은 몰 TOP {n} (기간 {title_range})")
        print(df_to_string_kr(out.head(n), index=False))
        print()
        return True

    if intent == "Q5_RANGE_SUMMARY":
        if not e.mall:
            print("쇼핑몰명을 인식 못했어요. (엑셀에 존재하는 mall_name을 질문에 포함)\n")
            return True
        print_date_clamp_notice(qc, app.today)
        out_df = q5_range_summary(app.df_pc_all, d.resolved_start, d.resolved_end, e.mall, manufacturer=e.manufacturer)
        if out_df is None or len(out_df) == 0:
            print("기간 내 데이터가 없습니다.\n")
            return True
        scope = f", 제조사={e.manufacturer}" if e.manufacturer else ""
        print(f"\n[정형결과] {e.mall}보다 싼 몰별 위반건수 (날짜별) ({d.resolved_start} ~ {d.resolved_end}{scope})")
        print_result_any(out_df, output_dir=app.args.output_dir, prefix=f"Q5_RANGE_SUMMARY_{e.mall}_{d.resolved_start}_to_{d.resolved_end}")
        print()
        return True

    if intent == "Q5_RANGE_DETAIL":
        if not e.mall:
            print("쇼핑몰명을 인식 못했어요. (엑셀에 존재하는 mall_name을 질문에 포함)\n")
            return True
        print_date_clamp_notice(qc, app.today)
        out_df = q5_range_detail(
            app.df_pc_all,
            d.resolved_start,
            d.resolved_end,
            e.mall,
            manufacturer=e.manufacturer,
            include_target_price=True,
            include_diff=True,
            limit=HARD_LIMIT_ROWS if qc.slots.wants_all_rows else 5000,
        )
        if out_df is None or len(out_df) == 0:
            print("기간 내 데이터가 없습니다.\n")
            return True
        scope = f", 제조사={e.manufacturer}" if e.manufacturer else ""
        product_count = int(out_df["product_key"].nunique()) if "product_key" in out_df.columns else 0
        case_count = int(len(out_df))
        print(f"\n[정형결과] {e.mall}보다 싼 곳이 있는 상품 상세 (날짜별) ({d.resolved_start} ~ {d.resolved_end}{scope}) — 제품 {product_count}개 / 케이스 {case_count}행")
        print_result_any(out_df, output_dir=app.args.output_dir, prefix=f"Q5_RANGE_DETAIL_{e.mall}_{d.resolved_start}_to_{d.resolved_end}")
        print()
        return True

    if intent == "Q5":
        if not e.mall:
            print("쇼핑몰명을 인식 못했어요. (엑셀에 존재하는 mall_name을 질문에 포함)\n")
            return True
        include_target_price = wants_target_price(q)
        include_diff = wants_diff(q)
        require_target_price = include_target_price or include_diff

        effective_batch_date = d.batch_date
        out = q5_mall_not_cheapest(
            app.df_pc_all,
            effective_batch_date,
            e.mall,
            manufacturer=e.manufacturer,
            include_target_price=include_target_price,
            include_diff=include_diff,
            require_target_price=require_target_price,
        )
        if not out:
            print("데이터를 못 찾았어요.\n")
            return True

        scope = f"(제조사={e.manufacturer}) " if e.manufacturer else ""
        print(
            f"\n[정형결과] {effective_batch_date} {scope}모든 과자 중 '{out['target_mall']}'이(가) 최저가가 아닌 제품: "
            f"제품 {out['product_count']}개 / 케이스 {out['case_count']}건 / 전체 {out['total_products']}개"
        )

        table: pd.DataFrame = out["table"]
        if table.empty:
            print("(해당 없음)")
        else:
            print(df_to_string_kr(table, index=False))

        ensure_dir(app.args.output_dir)
        mtag = e.manufacturer if e.manufacturer else "ALL"
        path = os.path.join(app.args.output_dir, f"Q5_not_cheapest_{out['target_mall']}_{mtag}_{out['batch_date']}.csv")
        table.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n[저장] CSV: {path}\n")
        return True

    if intent == "Q6":
        if not e.mall:
            print("쇼핑몰명을 인식 못했어요. (엑셀에 존재하는 mall_name을 질문에 포함)\n")
            return True
        out = q6_mall_is_cheapest(app.df_pc_all, d.batch_date, e.mall, manufacturer=e.manufacturer)
        if not out:
            print("데이터를 못 찾았어요.\n")
            return True
        scope = f"(제조사={e.manufacturer}) " if e.manufacturer else ""
        print(
            f"\n[정형결과] {d.batch_date} {scope}'{e.mall}'이(가) 최저가인 제품: "
            f"제품 {out['product_count']}개 / 케이스 {out['case_count']}건 / 전체 {out['total_products']}개"
        )
        table = out["table"]
        if table.empty:
            print("(해당 없음)")
        else:
            print(df_to_string_kr(table, index=False))
        print()
        return True

    if intent == "Q7":
        if not e.mall:
            print("쇼핑몰명을 인식 못했어요. (엑셀에 존재하는 mall_name을 질문에 포함)\n")
            return True
        n = parse_top_n(q, default_n=10)
        metric = parse_metric_kor(q)
        ascending = parse_sort_direction(q, metric)

        out = q7_mall_metric_topn(app.df_pc_all, d.batch_date, e.mall, metric, n, ascending, manufacturer=e.manufacturer)
        if not out:
            print("데이터를 못 찾았어요(지표/날짜/몰/제조사 확인).\n")
            return True

        direction = "낮은값 TOP" if ascending else "높은값 TOP"
        scope = f"(제조사={e.manufacturer}) " if e.manufacturer else ""
        print(f"\n[정형결과] {d.batch_date} {scope}{e.mall} {metric} {direction} {n}개")
        print(df_to_string_kr(out["table"], index=False))
        print()
        return True

    if intent == "Q8_TREND":
        if not e.product_key:
            print("제품을 인식 못했어요. (엑셀에 존재하는 제품코드/제품명이 질문에 포함되어야 함)\n")
            return True
        print_date_clamp_notice(qc, app.today)
        out_df = q8_product_trend(app.df_pc_all, product_key=e.product_key, start_date=d.resolved_start, end_date=d.resolved_end, manufacturer=e.manufacturer, mall=e.mall)
        if out_df is None:
            print("기간 내 데이터가 없습니다.\n")
            return True
        mall_scope = f", 몰={e.mall}" if e.mall else ""
        manu_scope = f", 제조사={e.manufacturer}" if e.manufacturer else ""
        title_range = d.resolved_start if not d.is_range else f"{d.resolved_start} ~ {d.resolved_end}"
        print(f"\n[정형결과] {e.product_key} 가격 추이 ({title_range}{manu_scope}{mall_scope})")
        print_result_any(out_df, output_dir=app.args.output_dir, prefix=f"Q8_trend_{e.product_key}_{d.resolved_start}_to_{d.resolved_end}")
        print()
        return True

    if intent == "Q7B_MANU_RANGE_TOPN":
        if not e.manufacturer:
            print("제조사를 인식 못했어요. (엑셀에 존재하는 manufacturer 또는 자사/우리/당사 포함)\n")
            return True
        print_date_clamp_notice(qc, app.today)
        n = parse_top_n(q, default_n=20)
        metric = parse_metric_kor(q)
        ascending = parse_sort_direction(q, metric)
        out_df = q7b_manufacturer_range_topn(app.df_pc_all, d.resolved_start, d.resolved_end, e.manufacturer, metric, n, ascending)
        if out_df is None or len(out_df) == 0:
            print("기간 내 데이터가 없습니다.\n")
            return True
        direction = "낮은값 TOP" if ascending else "높은값 TOP"
        title_range = d.resolved_start if not d.is_range else f"{d.resolved_start} ~ {d.resolved_end}"
        print(f"\n[정형결과] {e.manufacturer} {metric} {direction} {n}개 ({title_range})")
        print(df_to_string_kr(out_df, index=False))
        print()
        return True

    if intent == "Q7C_MANU_RANGE_TOPN_WITH_DATE":
        if not e.manufacturer:
            print("제조사를 인식 못했어요. (엑셀에 존재하는 manufacturer 또는 자사/우리/당사 포함)\n")
            return True
        print_date_clamp_notice(qc, app.today)
        n = parse_top_n(q, default_n=20)
        metric = parse_metric_kor(q)
        ascending = parse_sort_direction(q, metric)
        out_df = q7c_manufacturer_range_topn_with_date(app.df_pc_all, d.resolved_start, d.resolved_end, e.manufacturer, metric, n, ascending)
        if out_df is None or len(out_df) == 0:
            print("기간 내 데이터가 없습니다.\n")
            return True
        direction = "낮은값 TOP" if ascending else "높은값 TOP"
        title_range = d.resolved_start if not d.is_range else f"{d.resolved_start} ~ {d.resolved_end}"
        print(f"\n[정형결과] {e.manufacturer} {metric} {direction} {n}개 - 날짜 포함 ({title_range})")
        print(df_to_string_kr(out_df, index=False))
        print()
        return True

    if intent == "Q7D_MANU_RANGE_TOPN_BY_MALL":
        if not e.manufacturer:
            print("제조사를 인식 못했어요. (엑셀에 존재하는 manufacturer 또는 자사/우리/당사 포함)\n")
            return True
        print_date_clamp_notice(qc, app.today)
        n = parse_top_n(q, default_n=5)
        metric = parse_metric_kor(q)
        ascending = parse_sort_direction(q, metric)
        out_map = q7d_manufacturer_range_topn_by_mall(app.df_pc_all, d.resolved_start, d.resolved_end, e.manufacturer, metric, n, ascending)
        if not out_map:
            print("기간 내 데이터가 없습니다.\n")
            return True
        direction = "낮은값 TOP" if ascending else "높은값 TOP"
        title_range = d.resolved_start if not d.is_range else f"{d.resolved_start} ~ {d.resolved_end}"
        print(f"\n[정형결과] {e.manufacturer} {metric} {direction} - 몰별 TOP {n} ({title_range})")
        for mall_name in sorted(out_map.keys()):
            print(f"\n--- {mall_name} ---")
            print(df_to_string_kr(out_map[mall_name], index=False))
        print()
        return True

    if intent == "Q9_MALL_SUMMARY_TABLE":
        if not e.mall:
            print("쇼핑몰명을 인식 못했어요. (엑셀에 존재하는 mall_name을 질문에 포함)\n")
            return True
        print_date_clamp_notice(qc, app.today)
        out_df = q9_mall_summary_table(app.df_pc_all, d.resolved_start, d.resolved_end, e.mall, manufacturer=e.manufacturer)
        if out_df is None or len(out_df) == 0:
            print("기간 내 데이터가 없습니다.\n")
            return True
        title_range = d.resolved_start if not d.is_range else f"{d.resolved_start} ~ {d.resolved_end}"
        scope = f", 제조사={e.manufacturer}" if e.manufacturer else ""
        print(f"\n[정형결과] {e.mall} 요약 가격표 ({title_range}{scope})")
        print_result_any(out_df, output_dir=app.args.output_dir, prefix=f"Q9_summary_{e.mall}_{d.resolved_start}_to_{d.resolved_end}")
        print()
        return True

    return False


# ============================================================
# Fallback executor
# ============================================================
def execute_llm_fallback(app: AppContext, qc: QueryContext):
    q = qc.question
    e = qc.entities
    d = qc.dates

    if looks_like_product_query(q) and e.product_key is None and any(k in q for k in PRODUCT_QUERY_HINTS):
        print("제품을 인식 못했어요. 현재 데이터에 없는 제품명/제품코드일 수 있습니다.\n")
        return

    if should_block_plan_fallback(q, qc.slots):
        print("정형 규칙에 없는 위반/비교 질의입니다. 정형 라우터 보강이 필요합니다.\n")
        return

    if not (app.args.enable_llm and not app.args.disable_llm_plan):
        print("규칙에 없는 질문입니다. (LLM SAFE 폴백 비활성화)\n")
        return

    print("[LLM SAFE 폴백] PandasPlan JSON plan 생성 → 실행 중...")
    try:
        entities = {
            "pkey": e.product_key,
            "mall": e.mall,
            "mall_list": e.mall_list,
            "manu": e.manufacturer,
            "manu_list": e.manufacturer_list,
            "batch_date": d.batch_date,
        }
        llm_plan_ctx = build_llm_plan_context(
            app.df_pc_all, app.df_data_all, app.today, q, entities, asdict(qc.slots), d.resolved_start, d.resolved_end
        )

        try:
            raw_plan = llm_generate_plan(
                q,
                llm_plan_ctx,
                model_path=app.args.llm_model_path,
                n_ctx=app.args.llm_n_ctx,
                max_tokens=app.args.llm_max_tokens,
                temperature=app.args.llm_temperature,
                n_threads=app.args.llm_threads,
                n_gpu_layers=app.args.llm_gpu_layers,
            )
            if app.args.print_plan:
                print("\n[RAW PLAN]")
                print(json.dumps(raw_plan, ensure_ascii=False, indent=2))
        except Exception as ex:
            print(f"[WARN] LLM JSON 생성 실패 → 규칙 기반 mini-plan으로 대체: {repr(ex)}")
            raw_plan = build_rule_based_miniplan(
                q,
                batch_date=d.batch_date,
                start_date=d.resolved_start,
                end_date=d.resolved_end,
                mall=e.mall,
                manu=e.manufacturer,
                pkey=e.product_key,
                mall_list=e.mall_list,
                manu_list=e.manufacturer_list,
            )
            if app.args.print_plan:
                print("\n[RULE-BASED RAW PLAN]")
                print(json.dumps(raw_plan, ensure_ascii=False, indent=2))

        repaired_plan = repair_plan_structure(
            raw_plan,
            question=q,
            batch_date=d.batch_date,
            start_date=d.resolved_start,
            end_date=d.resolved_end,
            mall=e.mall,
            manu=e.manufacturer,
            pkey=e.product_key,
            mall_list=e.mall_list,
            manu_list=e.manufacturer_list,
        )
        repaired_plan = enrich_plan_semantics(
            repaired_plan,
            question=q,
            start_date=d.resolved_start,
            end_date=d.resolved_end,
            mall=e.mall,
            manu=e.manufacturer,
            pkey=e.product_key,
            mall_list=e.mall_list,
            manu_list=e.manufacturer_list,
        )
        repaired_plan = validate_plan_against_question(
            q,
            repaired_plan,
            mall=e.mall,
            manu=e.manufacturer,
            pkey=e.product_key,
            mall_list=e.mall_list,
            manu_list=e.manufacturer_list,
            start_date=d.resolved_start,
            end_date=d.resolved_end,
        )

        if app.args.print_plan:
            print("\n[REPAIRED RAW PLAN]")
            print(json.dumps(repaired_plan, ensure_ascii=False, indent=2))

        plan = validate_and_normalize_plan(repaired_plan, df_pc_all=app.df_pc_all, df_data_all=app.df_data_all)

        if app.args.print_plan:
            print_normalized_plan(plan)

        out_df = execute_plan(plan, df_pc_all=app.df_pc_all, df_data_all=app.df_data_all)
        print("\n[LLM SAFE 결과]")
        print_result_any(out_df, output_dir=app.args.output_dir, prefix="LLM_SAFE")
        print()

    except Exception as ex:
        print("LLM SAFE 폴백 실패:", repr(ex))
        print("힌트: 질문에 날짜/몰/지표/제조사(또는 자사/우리/당사)를 포함하면 성공률이 더 올라갑니다.\n")


# ============================================================
# Bootstrap
# ============================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Snacks POC CLI Router Redesigned")
    p.add_argument("--folder", default="/home/siwasoft/gsllm/xlsx_input", help="xlsx folder path")
    p.add_argument("--file-regex", default=r"^\d{4}-\d{2}-\d{2}_snacks.*\.xlsx$", help="xlsx basename regex")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="output directory for CSV")

    p.add_argument("--enable-chroma", action="store_true", help="enable chroma upsert")
    p.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH)
    p.add_argument("--coll-summary", default=DEFAULT_COLL_SUMMARY)
    p.add_argument("--coll-reviews", default=DEFAULT_COLL_REVIEWS)
    p.add_argument("--disable-reviews-digest", action="store_true", help="disable reviews digest upsert")

    p.add_argument("--enable-llm", action="store_true", help="enable LLM fallback (intent-only + plan)")
    p.add_argument("--llm-model-path", default=DEFAULT_LLM_MODEL_PATH)
    p.add_argument("--llm-n-ctx", type=int, default=DEFAULT_LLM_N_CTX)
    p.add_argument("--llm-max-tokens", type=int, default=DEFAULT_LLM_MAX_TOKENS)
    p.add_argument("--llm-temperature", type=float, default=DEFAULT_LLM_TEMPERATURE)
    p.add_argument("--llm-threads", type=int, default=DEFAULT_LLM_THREADS)
    p.add_argument("--llm-gpu-layers", type=int, default=DEFAULT_LLM_GPU_LAYERS)

    p.add_argument("--trace-route", action="store_true", help="print route decisions")
    p.add_argument("--print-plan", action="store_true", help="print RAW/RULE-BASED/REPAIRED/NORMALIZED plan")
    p.add_argument("--disable-llm-plan", action="store_true", help="disable PandasPlan fallback")

    p.set_defaults(
        enable_llm=DEFAULT_ENABLE_LLM,
        print_plan=DEFAULT_PRINT_PLAN,
    )
    return p


def bootstrap_app(args: argparse.Namespace) -> AppContext:
    files = discover_xlsx_files(args.folder, args.file_regex)
    if not files:
        raise RuntimeError(f"xlsx 파일을 찾지 못했습니다. folder={args.folder}, regex={args.file_regex}")

    df_data_all, df_pc_all = load_excels_multi(args.folder, files)

    if len(df_data_all) == 0 or len(df_pc_all) == 0:
        raise RuntimeError("로드된 데이터가 비어 있습니다.")

    df_data_all["batch_date"] = normalize_batch_date_series(df_data_all["batch_date"])
    df_pc_all["batch_date"] = normalize_batch_date_series(df_pc_all["batch_date"])

    catalog = build_catalog(df_pc_all, df_data_all)
    today = today_from_df(df_pc_all)
    default_start = min_date_from_df(df_pc_all, fallback=today)

    return AppContext(
        args=args,
        df_data_all=df_data_all,
        df_pc_all=df_pc_all,
        catalog=catalog,
        today=today,
        default_start=default_start,
    )


# ============================================================
# Main
# ============================================================
def main():
    args = build_argparser().parse_args()

    print("=== Snacks POC CLI Router Redesigned ===")
    print(f"(자사 제조사 고정) MY_MANUFACTURER = {MY_MANUFACTURER}")
    print(f"[DEFAULT LLM MODEL] {args.llm_model_path}")
    print(f"[DEFAULT LLM CTX] n_ctx={args.llm_n_ctx}, gpu_layers={args.llm_gpu_layers}, print_plan={args.print_plan}, enable_llm={args.enable_llm}")

    if rf_process is None:
        print("[FUZZY] rapidfuzz 미설치: mall/manufacturer 유사매칭 비활성화")
    else:
        print(f"[FUZZY] rapidfuzz 활성화(cutoff={FUZZY_CUTOFF})")

    try:
        app = bootstrap_app(args)
    except Exception as e:
        print("초기화 실패:", repr(e))
        return

    print(f"로드 완료: DATA={len(app.df_data_all):,}행, PRICE_COMPARE={len(app.df_pc_all):,}행")
    print(f"날짜 범위: {app.df_pc_all['batch_date'].min()} ~ {app.df_pc_all['batch_date'].max()}")
    print(f"[카탈로그] 몰={len(app.catalog['malls'])}개, 제조사={len(app.catalog['manufacturers'])}개, 제품={len(app.catalog['product_key_set'])}개")

    if args.enable_chroma:
        try:
            col_sum, col_rev = get_chroma_collections(args.chroma_path, args.coll_summary, args.coll_reviews)
            upsert_price_compare(col_sum, app.df_pc_all, version="redesigned_v1")
            if not args.disable_reviews_digest:
                upsert_reviews_digest(col_rev, app.df_data_all, version="redesigned_v1")
            print("Chroma 업서트 완료")
        except Exception as e:
            print("Chroma 오류(무시하고 계속):", repr(e))

    print(f"\n현재 합본 기준 today(batch_date) = {app.today} (default_start={app.default_start})\n")

    if args.enable_llm:
        print("[LLM] intent-only 폴백 활성화 (UNKNOWN일 때 intent 분류 시도)")
        if not args.disable_llm_plan:
            print("[LLM] PandasPlan SAFE 폴백도 활성화\n")
        else:
            print("[LLM] PandasPlan SAFE 폴백 비활성화(intent-only만)\n")
    else:
        print("[LLM] 비활성화\n")

    print("질문 입력. 종료: exit\n")

    while True:
        q_raw = input("Q> ").strip()
        if q_raw.lower() in ["exit", "quit"]:
            break
        if not q_raw:
            continue

        qc = build_query_context(app, q_raw)
        plan_allowed = not should_block_plan_fallback(qc.question, qc.slots)
        print_parsed_debug(qc, plan_allowed=plan_allowed)

        if args.enable_llm and qc.intent == "UNKNOWN":
            try:
                intent2 = llm_classify_intent(
                    qc.question,
                    today=app.today,
                    default_start=app.default_start,
                    model_path=args.llm_model_path,
                    n_ctx=args.llm_n_ctx,
                    max_tokens=min(240, args.llm_max_tokens),
                    temperature=max(0.0, min(0.2, args.llm_temperature)),
                    n_threads=args.llm_threads,
                    n_gpu_layers=args.llm_gpu_layers,
                    mall=qc.entities.mall,
                    pkey=qc.entities.product_key,
                    manu=qc.entities.manufacturer,
                )
                if intent2 != "UNKNOWN":
                    qc.intent = intent2
            except Exception:
                pass

        if args.trace_route:
            print(
                f"[ROUTE] intent={qc.intent} pkey={qc.entities.product_key} mall={qc.entities.mall} manu={qc.entities.manufacturer} "
                f"batch_date={qc.dates.batch_date} start_date={qc.dates.resolved_start} end_date={qc.dates.resolved_end}"
            )

        handled = execute_structured_route(app, qc)
        if not handled:
            execute_llm_fallback(app, qc)

        gc.collect()


if __name__ == "__main__":
    main()