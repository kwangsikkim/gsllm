#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
poc_snacks_cli_router_v3.2.py

v3.3.2-full
- 기존 v3.3.x 기반
- 추가 반영
  1) "싼 것들 / 비싼 것들 / 높은 것들 / 낮은 것들" 류는 summary(df_pc_all) 우선
  2) "몰별로 / 쇼핑몰별로 / 몰마다 / 쇼핑몰마다 / 묶어서" 슬롯 추가
  3) 제조사 + 기간 + 가격지표 + 몰별 grouping 질의용 정형 intent 추가
     -> Q7D_MANU_RANGE_TOPN_BY_MALL
  4) raw 질문 limit 정책 유지
  5) TopN/랭킹형 기본 select에 manufacturer, mall_name 포함

주의
- CSV/DF 원본 컬럼은 영문 유지
- 터미널 출력에서만 한글 컬럼명 치환
- rapidfuzz 미설치 시 자동 비활성
- llama_cpp 미설치 시 --enable-llm 사용 불가
"""

import os
import re
import json
import gc
import argparse
import calendar
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Optional, Tuple, Dict, List, Literal

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

SHIP_WORDS = ["배송", "지연", "늦", "파손", "오배송", "택배", "배송비"]
SHIP_NEG_WORDS = ["불만", "문제", "이슈", "클레임", "컴플", "항의", "불편", "민원"]

DEFAULT_OUTPUT_DIR = "./outputs"

MY_MANUFACTURER = "해태제과"
MY_WORDS = [
    "자사", "우리", "당사",
    "우리회사", "우리 회사", "본사",
    "우리제품", "우리 제품", "당사제품", "당사 제품",
]

DEFAULT_ENABLE_LLM = False
DEFAULT_LLM_MODEL_PATH = "/home/siwasoft/samoo/mcp/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
DEFAULT_LLM_N_CTX = 2048
DEFAULT_LLM_MAX_TOKENS = 700
DEFAULT_LLM_TEMPERATURE = 0.15
DEFAULT_LLM_THREADS = 16
DEFAULT_LLM_GPU_LAYERS = 0

MAX_RESULT_ROWS_PRINT = 2000
MAX_RESULT_COLS_PRINT = 200
CSV_SAVE_THRESHOLD = 600
HARD_LIMIT_ROWS = 20000

DEFAULT_RAW_LIMIT = 300
DEFAULT_SUMMARY_LIMIT = 50

FUZZY_CUTOFF = 88


# ============================================================
# Text normalization helpers
# ============================================================
DASH_VARIANTS = ["–", "—", "−"]
WAVE_VARIANTS = ["∼", "〜"]


def normalize_range_separators(q: str) -> str:
    t = q
    for d in DASH_VARIANTS:
        t = t.replace(d, "-")
    for w in WAVE_VARIANTS:
        t = t.replace(w, "~")
    t = re.sub(r"\bto\b", "-", t, flags=re.IGNORECASE)
    return t


# ============================================================
# Terminal column header translation (EN -> KO)
# ============================================================
COLNAME_KR: Dict[str, str] = {
    "batch_date": "배치일자",
    "manufacturer": "제조사",
    "mall_name": "쇼핑몰",
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
    "cheaper_mall": "더저렴한몰",
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
}


def df_for_terminal(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return df.rename(columns=COLNAME_KR).copy()
    except Exception:
        return df.copy()


def df_to_string_kr(df: pd.DataFrame, index: bool = False) -> str:
    return df_for_terminal(df).to_string(index=index)


# ============================================================
# Utils
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def now_ts() -> int:
    return int(datetime.now().timestamp())


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


def has_my_words(q: str) -> bool:
    return any(w in q for w in MY_WORDS)


# ============================================================
# Fuzzy matching helpers
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
        inferred_year = None
    if inferred_year is not None:
        mm = re.search(r"\b(\d{1,2})\s*[-/]\s*(\d{1,2})\b", q)
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

    dates: List[str] = []
    for mm in re.finditer(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", q):
        y, mo, d = int(mm.group(1)), int(mm.group(2)), int(mm.group(3))
        dates.append(f"{y:04d}-{mo:02d}-{d:02d}")

    for mm in re.finditer(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", q):
        y, mo, d = int(mm.group(1)), int(mm.group(2)), int(mm.group(3))
        dates.append(f"{y:04d}-{mo:02d}-{d:02d}")

    seen = set()
    uniq: List[str] = []
    for d in dates:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def parse_date_range(question: str, default_end: str, default_start: str) -> Tuple[str, str]:
    q = normalize_range_separators(question.strip())

    try:
        dt_end = datetime.strptime(str(default_end)[:10], "%Y-%m-%d")
    except Exception:
        dt_end = datetime.now()

    if any(k in q for k in ["이번주", "이번 주", "금주"]):
        s, e = _week_range_for(dt_end)
        return max(s, default_start), min(e, default_end)

    if any(k in q for k in ["지난주", "지난 주", "저번주", "저번 주", "전주"]):
        base = dt_end - timedelta(days=7)
        s, e = _week_range_for(base)
        return max(s, default_start), min(e, default_end)

    if any(k in q for k in ["이번달", "이번 달", "금월"]):
        s, e = _month_range_for(dt_end)
        return max(s, default_start), min(e, default_end)

    if any(k in q for k in ["지난달", "지난 달", "저번달", "저번 달", "전월"]):
        y, m = dt_end.year, dt_end.month
        if m == 1:
            y, m = y - 1, 12
        else:
            m -= 1
        dt_prev = datetime(y, m, 15)
        s, e = _month_range_for(dt_prev)
        return max(s, default_start), min(e, default_end)

    m = re.search(r"(최근|지난)\s*([0-9]{1,3})\s*일(간|동안|일간)?", q)
    if m:
        n = max(1, min(int(m.group(2)), 365))
        end = default_end
        start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=n - 1)).strftime("%Y-%m-%d")
        if start < default_start:
            start = default_start
        return start, end

    dl = parse_dates_list(q, default_date=default_end)
    if len(dl) >= 2:
        s, e = dl[0], dl[1]
        if s > e:
            s, e = e, s
        return max(s, default_start), min(e, default_end)

    single = parse_batch_date(q, default_date=default_end)
    return max(single, default_start), min(single, default_end)


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
# Catalog / Parsing
# ============================================================
def build_catalog(df_pc: pd.DataFrame, df_data: pd.DataFrame) -> Dict[str, Any]:
    malls = sorted(set(pd.concat([df_pc["mall_name"], df_data["mall_name"]]).astype(str)))
    manufacturers = sorted(set(pd.concat([df_pc["manufacturer"], df_data["manufacturer"]]).astype(str)))
    product_keys = sorted(set(pd.concat([df_pc["product_key"], df_data["product_key"]]).astype(str)))
    name_to_key = dict(zip(df_pc["product_name"].astype(str), df_pc["product_key"].astype(str)))
    key_to_manufacturer = dict(zip(df_pc["product_key"].astype(str), df_pc["manufacturer"].astype(str)))
    return {
        "malls": malls,
        "manufacturers": manufacturers,
        "product_key_set": set(product_keys),
        "name_to_key": name_to_key,
        "key_to_manufacturer": key_to_manufacturer,
    }


def extract_target_mall(question: str, malls: List[str]) -> Optional[str]:
    q = question.strip()
    for m in sorted(malls, key=len, reverse=True):
        if m and m in q:
            return m
    return None


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


def resolve_my_manufacturer(question: str, manufacturers: List[str], fallback_my: str = MY_MANUFACTURER) -> Optional[str]:
    if has_my_words(question):
        return fallback_my
    return None


def parse_product_key(question: str, catalog: Dict[str, Any]) -> Optional[str]:
    q = question.strip()
    keyset = catalog["product_key_set"]
    name_to_key: Dict[str, str] = catalog["name_to_key"]

    if len(name_to_key) <= 5000:
        for pname in sorted(name_to_key.keys(), key=len, reverse=True):
            if pname and pname in q:
                pk = name_to_key.get(pname)
                if pk in keyset:
                    return pk

    candidates: List[str] = []
    for m in re.finditer(r"\b([A-Za-z])\s*0*([0-9]{1,4})\b", q):
        prefix = m.group(1).upper()
        num = int(m.group(2))
        candidates.append(f"{prefix}{num:03d}")

    m2 = re.search(r"([A-Za-z가-힣]+)\s*과자\s*0*([0-9]{1,4})", q)
    if m2:
        head = m2.group(1)
        num = int(m2.group(2))
        cand_name = f"{head}과자{num:03d}"
        if cand_name in name_to_key:
            pk = name_to_key[cand_name]
            if pk in keyset:
                return pk

    for cand in candidates:
        if cand in keyset:
            return cand

    return None


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
        "전체 데이터 말고", "원본 말고", "raw 말고"
    ])


def prefers_raw_semantic(q: str) -> bool:
    return any(k in q for k in [
        "원본", "raw", "오퍼", "오퍼 목록", "전건", "원자료"
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
        "싼 애들", "비싼 애들", "저렴한 애들"
    ])


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
    if "min_price" in q:
        return "min_price"
    if "avg_price" in q:
        return "avg_price"
    if "max_price" in q:
        return "max_price"
    if ("평균" in q) or ("avg" in q.lower()):
        return "avg_price"
    if any(k in q for k in ["최저가", "최저", "저렴", "싼", "싸게", "제일 싼", "가장 싼", "가장 저렴"]):
        return "min_price"
    if any(k in q for k in ["최고가", "최대", "비싼", "높은", "큰값", "제일 비싼", "가장 비싼"]):
        return "max_price"
    # generic price bucket words -> 보통 싼 것들/비싼 것들
    if any(k in q for k in ["싼 것들", "저렴한 것들", "싼 상품", "저렴한 상품"]):
        return "min_price"
    if any(k in q for k in ["비싼 것들", "높은 것들", "비싼 상품", "높은 상품"]):
        return "max_price"
    return "min_price"


def parse_sort_direction(question: str, metric: str) -> bool:
    q = question.strip()
    if any(k in q for k in ["높", "비싼", "최고", "큰", "비싸게", "가장 비싼", "높은 순", "내림차순"]):
        return False
    if any(k in q for k in ["낮", "저렴", "싼", "낮은 순", "오름차순"]):
        return True
    if metric == "max_price":
        return False
    return True


VIOLATION_WORDS = [
    "위반", "정책 위반", "규정 위반", "컴플", "컴플라이언스", "페널티",
    "보상", "환급", "배상", "물어내", "물어 내"
]
DETAIL_WORDS_EXT = [
    "상세", "내역", "목록", "리스트", "전부", "전체", "원본", "보여줘",
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
        "싼 것들", "비싼 것들", "높은 것들", "낮은 것들"
    ])


def _has_range_semantics(q: str) -> bool:
    return any(k in q for k in [
        "~", "-", "부터", "까지", "최근", "지난",
        "이번주", "이번 주", "지난주", "지난 주",
        "이번달", "이번 달", "지난달", "지난 달",
        "금주", "금월", "전월"
    ])


def _is_not_cheapest_semantic(q: str) -> bool:
    has_price_compare = any(k in q for k in [
        "최저가", "가장 싸", "가장 저렴", "싼", "저렴", "더 싸", "더 저렴", "제일 싼"
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
        or (has_price_compare and any(k in q for k in ["아닌", "아니다", "아닙"]))
    )


def _is_cheapest_only_semantic(q: str) -> bool:
    return (
        any(k in q for k in ["최저가", "제일 싼", "가장 싼", "가장 저렴"])
        and any(k in q for k in ["만", "것만", "제품만", "과자만"])
        and ("아닌" not in q)
    )


def extract_semantic_slots(q: str) -> Dict[str, Any]:
    q = q.strip()
    return {
        "has_violation": _has_violation_semantics(q),
        "wants_detail": _has_detail_semantics(q),
        "wants_trend": _has_trend_semantics(q),
        "wants_offer": _has_offer_semantics(q),
        "wants_price": _has_price_semantics(q),
        "wants_topn": _has_topn_semantics(q),
        "wants_shipping_issue": _has_shipping_issue_semantics(q),
        "has_metric": _has_metric_semantics(q),
        "has_range": _has_range_semantics(q),
        "not_cheapest": _is_not_cheapest_semantic(q),
        "cheapest_only": _is_cheapest_only_semantic(q),
        "wants_date_in_rows": wants_date_in_rows(q),
        "prefers_summary": prefers_summary_semantic(q),
        "prefers_raw": prefers_raw_semantic(q),
        "wants_all_rows": wants_all_rows(q),
        "wants_group_by_mall": wants_group_by_mall(q),
        "has_generic_price_bucket_words": has_generic_price_bucket_words(q),
    }


# ============================================================
# Intent detection
# ============================================================
def detect_intent(q: str, has_product: bool, has_mall: bool, has_manu: bool) -> str:
    s = extract_semantic_slots(q)

    if has_mall and s["has_violation"]:
        if s["wants_trend"]:
            return "A1_VIOL_TREND"
        return "A1_VIOL_DETAIL"

    if has_product and s["wants_offer"]:
        return "Q1_DETAIL"

    if has_mall and s["not_cheapest"]:
        return "Q5"

    if has_mall and s["cheapest_only"]:
        return "Q6"

    if has_mall and s["wants_topn"] and s["has_metric"]:
        return "Q7"

    if has_product and s["wants_trend"]:
        return "Q8_TREND"

    if has_product and s["wants_price"]:
        return "Q1"

    if s["wants_shipping_issue"]:
        return "Q4"

    # 제조사 + 기간 + 가격지표
    if has_manu and s["has_metric"] and s["has_range"] and (not has_mall) and (not has_product):
        if s["wants_group_by_mall"]:
            return "Q7D_MANU_RANGE_TOPN_BY_MALL"
        if s["wants_date_in_rows"]:
            return "Q7C_MANU_RANGE_TOPN_WITH_DATE"
        return "Q7B_MANU_RANGE_TOPN"

    # 몰 + 가격지표 + generic bucket words
    if has_mall and s["has_metric"] and (s["wants_topn"] or s["has_generic_price_bucket_words"]) and (not has_product):
        return "Q7"

    if has_mall and s["prefers_summary"] and (not has_product):
        return "Q9_MALL_SUMMARY_TABLE"

    return "UNKNOWN"


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


def upsert_price_compare(col_sum, df_pc: pd.DataFrame, batch_size: int = 500, version: str = "v3_3_2_full"):
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


def upsert_reviews_digest(col_rev, df_data: pd.DataFrame, batch_size: int = 500, version: str = "v3_3_2_full"):
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

    cols = ["batch_date", "manufacturer", "mall_name", "product_key", "product_name"]
    for c in ["rank", "price", "url", "item_name", "comments_top5", "date"]:
        if c in cur.columns and c not in cols:
            cols.append(c)

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
            "count": 0,
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
        "count": int(out["product_key"].nunique()),
        "table": out,
    }


def q6_mall_is_cheapest(df_pc: pd.DataFrame, batch_date: str, target_mall: str, manufacturer: Optional[str] = None):
    cur = df_pc[df_pc["batch_date"] == batch_date].copy()
    if manufacturer:
        cur = cur[cur["manufacturer"] == manufacturer].copy()
    if len(cur) == 0:
        return None

    best_idx = cur.groupby("product_key")["min_price"].idxmin()
    best_rows = cur.loc[best_idx].copy()

    hit = best_rows[best_rows["mall_name"] == target_mall].copy()
    out = hit[["manufacturer", "product_key", "product_name", "mall_name", "min_price"]].rename(
        columns={"mall_name": "cheapest_mall", "min_price": "cheapest_price"}
    ).sort_values(["product_key"], ascending=True)

    return {
        "batch_date": batch_date,
        "target_mall": target_mall,
        "manufacturer": manufacturer,
        "total_products": int(best_rows["product_key"].nunique()),
        "count": int(out["product_key"].nunique()),
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

    out = cur[["manufacturer", "mall_name", "product_key", "product_name", metric]].sort_values(metric, ascending=bool(ascending)).head(int(max(1, n))).copy()
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
    cur = cur.sort_values(["batch_date", "value"], ascending=[True, ascending]).head(int(max(1, n))).reset_index(drop=True)
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
# LLM helpers
# ============================================================
LLM = None


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
        '"intent": "Q1" | "Q1_DETAIL" | "Q4" | "Q5" | "Q6" | "Q7" | "Q8_TREND" | '
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
    if intent in ["A1_VIOL_DETAIL", "A1_VIOL_TREND", "Q5", "Q6", "Q7", "Q9_MALL_SUMMARY_TABLE"] and not mall:
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
        "Q1", "Q1_DETAIL", "Q4", "Q5", "Q6", "Q7", "Q8_TREND",
        "Q7B_MANU_RANGE_TOPN", "Q7C_MANU_RANGE_TOPN_WITH_DATE", "Q7D_MANU_RANGE_TOPN_BY_MALL",
        "Q9_MALL_SUMMARY_TABLE", "A1_VIOL_TREND", "A1_VIOL_DETAIL", "UNKNOWN"
    }
    if intent not in allow:
        return "UNKNOWN"

    if not validate_llm_intent(intent, mall=mall, pkey=pkey, manu=manu):
        return "UNKNOWN"
    return intent


# ============================================================
# SAFE PandasPlan
# ============================================================
PandasSource = Literal["df_pc_all", "df_data_all"]
FilterOp = Literal["=", "!=", "contains", "between"]
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
    # generic price bucket words는 summary 우선
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
        "- Generic cheap/expensive questions should prefer df_pc_all summary.\n"
        "- If prefers summary/price table/min/avg/max/topN/trend -> df_pc_all.\n"
        "- If raw/original/offers/detail -> df_data_all.\n"
        "- If entities.mall exists, add mall_name '=' filter.\n"
        "- If entities.manu exists, add manufacturer '=' filter.\n"
        "- If entities.pkey exists, add product_key '=' filter.\n"
        "- Use resolved_date_range as batch_date filter.\n"
        "- For df_data_all raw output, prefer sort by rank asc or price asc.\n"
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
        out = llm(prompt, max_tokens=int(max_tokens), temperature=float(temperature), stop=["```", "<|end|>", "<|start|>"])
        raw = out["choices"][0]["text"]
        last_raw = raw
        jtxt = _extract_json_object(raw)
        if jtxt:
            return json.loads(jtxt)

    raise ValueError(f"LLM output did not contain JSON object. raw={last_raw[:300]!r}")


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

    if mall:
        filters.append({"col": "mall_name", "op": "=", "value": mall})
    if manu:
        filters.append({"col": "manufacturer", "op": "=", "value": manu})
    if pkey:
        filters.append({"col": "product_key", "op": "=", "value": pkey})

    if source == "df_pc_all":
        limit = summary_limit_for_question(q)
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
        "select": ["batch_date", "mall_name", "manufacturer", "product_key", "product_name", "rank", "price", "date", "comments_top5"],
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

    if mall:
        _append_filter_if_missing(repaired["filters"], "mall_name", "=", value=mall)
    if manu:
        _append_filter_if_missing(repaired["filters"], "manufacturer", "=", value=manu)
    if pkey:
        _append_filter_if_missing(repaired["filters"], "product_key", "=", value=pkey)

    if not repaired["select"]:
        if repaired["source"] == "df_pc_all":
            repaired["select"] = ["batch_date", "manufacturer", "mall_name", "product_key", "product_name", "min_price", "avg_price", "max_price"]
        else:
            repaired["select"] = ["batch_date", "mall_name", "manufacturer", "product_key", "product_name", "rank", "price", "date", "comments_top5"]

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
) -> Dict[str, Any]:
    q = question.strip()
    metric = parse_metric_kor(q)

    if prefers_summary_semantic(q) or has_generic_price_bucket_words(q):
        plan["source"] = "df_pc_all"
    if prefers_raw_semantic(q) and not prefers_summary_semantic(q):
        plan["source"] = "df_data_all"

    if plan["source"] == "df_data_all":
        plan["limit"] = raw_limit_for_question(q)
        if not plan.get("sort"):
            plan["sort"] = [{"col": "rank", "dir": "asc"}]

    if plan["source"] == "df_pc_all":
        plan["limit"] = min(HARD_LIMIT_ROWS, int(plan.get("limit", summary_limit_for_question(q))))
        if not pkey and not plan.get("groupby"):
            if wants_date_in_rows(q):
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

    if (
        plan["source"] == "df_pc_all"
        and manu
        and start_date != end_date
        and not pkey
        and not wants_date_in_rows(q)
        and not wants_group_by_mall(q)
        and any(k in q for k in ["평균가", "평균", "최저가", "최고가", "상위", "하위", "높은", "낮은", "싼", "비싼"])
    ):
        agg_as = metric
        groupby_cols = ["manufacturer", "mall_name", "product_key", "product_name"]
        plan["groupby"] = groupby_cols
        plan["aggregations"] = [{"col": metric, "func": "mean", "as": agg_as}]
        plan["select"] = groupby_cols + [agg_as]
        plan["sort"] = [{"col": agg_as, "dir": "asc" if parse_sort_direction(q, metric) else "desc"}]
        plan["limit"] = summary_limit_for_question(q)

    return plan


def validate_plan_against_question(
    q: str,
    plan: Dict[str, Any],
    *,
    mall: Optional[str],
    manu: Optional[str],
    pkey: Optional[str],
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    filters = _normalize_list_or_empty(plan.get("filters"))

    def has_filter(col: str) -> bool:
        for f in filters:
            if isinstance(f, dict) and str(f.get("col") or "") == col:
                return True
        return False

    if mall and not has_filter("mall_name"):
        _append_filter_if_missing(filters, "mall_name", "=", value=mall)
    if manu and not has_filter("manufacturer"):
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
        plan["select"] = ["manufacturer", "mall_name", "product_key", "product_name", metric]
        plan["groupby"] = []
        plan["aggregations"] = []
        plan["sort"] = [{"col": metric, "dir": "asc" if parse_sort_direction(q, metric) else "desc"}]
        plan["limit"] = summary_limit_for_question(q)

    if prefers_summary_semantic(q):
        plan["source"] = "df_pc_all"
    if prefers_raw_semantic(q) and not prefers_summary_semantic(q):
        plan["source"] = "df_data_all"

    if plan["source"] == "df_data_all":
        plan["limit"] = raw_limit_for_question(q)
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
        if op not in ("=", "!=", "contains", "between"):
            continue
        if op == "between":
            start = f.get("start")
            end = f.get("end")
            if not start or not end:
                continue
            filters2.append(PandasFilter(col=col, op="between", start=str(start)[:10], end=str(end)[:10]))
        else:
            val = f.get("value")
            if op == "contains":
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
    for s in raw_sort[:10]:
        if not isinstance(s, dict):
            continue
        col = str(s.get("col") or "")
        direction = s.get("dir")
        if direction not in ("asc", "desc"):
            continue
        if col not in cols_set:
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


# ============================================================
# Plan block gate
# ============================================================
PLAN_FORBIDDEN_PATTERNS = [
    "위반", "보상", "배상", "정책 위반", "규정 위반", "컴플",
    "최저가 아닌", "최저가 아님", "더 싼", "더 저렴한",
    "최저가 놓친", "타겟몰보다 낮은", "다른 곳이 더 싸",
    "비교 위반", "물어내", "환급"
]


def should_block_plan_fallback(q: str) -> bool:
    return any(k in q for k in PLAN_FORBIDDEN_PATTERNS)


# ============================================================
# CLI
# ============================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Snacks POC CLI Router v3.3.2-full")
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
    return p


def main():
    args = build_argparser().parse_args()

    print("=== Snacks POC CLI Router v3.3.2-full ===")
    print(f"(자사 제조사 고정) MY_MANUFACTURER = {MY_MANUFACTURER}")
    if rf_process is None:
        print("[FUZZY] rapidfuzz 미설치: mall/manufacturer 유사매칭 비활성화")
    else:
        print(f"[FUZZY] rapidfuzz 활성화(cutoff={FUZZY_CUTOFF})")

    try:
        files = discover_xlsx_files(args.folder, args.file_regex)
        if not files:
            print("xlsx 파일을 찾지 못했습니다.")
            print(f"- folder: {args.folder}")
            print(f"- file_regex: {args.file_regex}")
            return
    except Exception as e:
        print("xlsx 파일 탐색 실패:", repr(e))
        return

    try:
        df_data_all, df_pc_all = load_excels_multi(args.folder, files)
    except Exception as e:
        print("멀티 엑셀 로드 실패:", repr(e))
        return

    if len(df_data_all) == 0 or len(df_pc_all) == 0:
        print("로드된 데이터가 비어 있습니다.")
        return

    df_data_all["batch_date"] = normalize_batch_date_series(df_data_all["batch_date"])
    df_pc_all["batch_date"] = normalize_batch_date_series(df_pc_all["batch_date"])

    print(f"로드 완료({len(files)}개 합본): DATA={len(df_data_all):,}행, PRICE_COMPARE={len(df_pc_all):,}행")
    print(f"날짜 범위: {df_pc_all['batch_date'].min()} ~ {df_pc_all['batch_date'].max()}")

    catalog = build_catalog(df_pc_all, df_data_all)
    print(f"[카탈로그] 몰={len(catalog['malls'])}개, 제조사={len(catalog['manufacturers'])}개, 제품={len(catalog['product_key_set'])}개")

    if args.enable_chroma:
        try:
            col_sum, col_rev = get_chroma_collections(args.chroma_path, args.coll_summary, args.coll_reviews)
            upsert_price_compare(col_sum, df_pc_all, version="v3_3_2_full")
            if not args.disable_reviews_digest:
                upsert_reviews_digest(col_rev, df_data_all, version="v3_3_2_full")
            print("Chroma 업서트 완료")
        except Exception as e:
            print("Chroma 오류(무시하고 계속):", repr(e))

    today = today_from_df(df_pc_all)
    default_start = min_date_from_df(df_pc_all, fallback=today)
    print(f"\n현재 합본 기준 today(batch_date) = {today} (default_start={default_start})\n")

    enable_llm = bool(args.enable_llm or DEFAULT_ENABLE_LLM)
    llm_plan_enabled = enable_llm and (not args.disable_llm_plan)

    if enable_llm:
        print("[LLM] intent-only 폴백 활성화 (UNKNOWN일 때 intent 분류 시도)")
        if llm_plan_enabled:
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

        q = normalize_range_separators(q_raw)

        pkey = parse_product_key(q, catalog)
        mall = extract_target_mall(q, catalog["malls"])
        manu = parse_manufacturer(q, catalog["manufacturers"])

        if mall is None:
            mall = _fuzzy_pick(q, catalog["malls"], cutoff=FUZZY_CUTOFF)
        if manu is None:
            manu = _fuzzy_pick(q, catalog["manufacturers"], cutoff=FUZZY_CUTOFF)

        my_manu = resolve_my_manufacturer(q, catalog["manufacturers"], fallback_my=MY_MANUFACTURER)
        if not manu and my_manu:
            manu = my_manu

        dates_list = parse_dates_list(q, default_date=today)
        batch_date = dates_list[0] if dates_list else parse_batch_date(q, default_date=today)
        start_date, end_date = parse_date_range(q, default_end=today, default_start=default_start)

        slots = extract_semantic_slots(q)
        intent = detect_intent(q, has_product=bool(pkey), has_mall=bool(mall), has_manu=bool(manu))

        print("[DEBUG:q]", q)
        print("[DEBUG:entities]", {"pkey": pkey, "mall": mall, "manu": manu, "batch_date": batch_date})
        print("[DEBUG:slots]", slots)
        print("[DEBUG:intent_before_llm]", intent)
        print("[DEBUG:block_plan]", should_block_plan_fallback(q))

        if enable_llm and intent == "UNKNOWN":
            try:
                intent2 = llm_classify_intent(
                    q,
                    today=today,
                    default_start=default_start,
                    model_path=args.llm_model_path,
                    n_ctx=args.llm_n_ctx,
                    max_tokens=min(220, args.llm_max_tokens),
                    temperature=max(0.0, min(0.2, args.llm_temperature)),
                    n_threads=args.llm_threads,
                    n_gpu_layers=args.llm_gpu_layers,
                    mall=mall,
                    pkey=pkey,
                    manu=manu,
                )
                if intent2 != "UNKNOWN":
                    intent = intent2
            except Exception:
                pass

        if args.trace_route:
            print(f"[ROUTE] intent={intent} pkey={pkey} mall={mall} manu={manu} batch_date={batch_date}")

        if intent in ["A1_VIOL_TREND", "A1_VIOL_DETAIL"] and not manu:
            manu = MY_MANUFACTURER

        if intent in ["Q1", "Q1_DETAIL", "Q8_TREND"] and not pkey:
            print("제품을 인식 못했어요. (엑셀에 존재하는 제품코드/제품명이 질문에 포함되어야 함)\n")
            continue
        if intent in ["Q5", "Q6", "Q7", "A1_VIOL_TREND", "A1_VIOL_DETAIL", "Q9_MALL_SUMMARY_TABLE"] and not mall:
            print("쇼핑몰명을 인식 못했어요. (엑셀에 존재하는 mall_name을 질문에 포함)\n")
            continue
        if intent in ["Q7B_MANU_RANGE_TOPN", "Q7C_MANU_RANGE_TOPN_WITH_DATE", "Q7D_MANU_RANGE_TOPN_BY_MALL"] and not manu:
            print("제조사를 인식 못했어요. (엑셀에 존재하는 manufacturer 또는 자사/우리/당사 포함)\n")
            continue

        # -----------------------------
        # 정형 처리
        # -----------------------------
        if intent == "A1_VIOL_TREND":
            out_df = a1_violation_trend(df_pc_all, start_date, end_date, manu, mall)
            if out_df is None or len(out_df) == 0:
                print("기간 내 데이터가 없습니다.\n")
                continue
            title_range = start_date if start_date == end_date else f"{start_date} ~ {end_date}"
            print(f"\n[A급 정형] {manu} '{mall}' 위반 추이 ({title_range})")
            print_result_any(out_df, output_dir=args.output_dir, prefix=f"A1_VIOL_TREND_{manu}_{mall}_{start_date}_to_{end_date}")
            print()
            continue

        if intent == "A1_VIOL_DETAIL":
            out_df = a1_violation_detail(df_pc_all, start_date, end_date, manu, mall, limit=5000)
            if out_df is None:
                print("기간 내 데이터가 없습니다.\n")
                continue
            title_range = start_date if start_date == end_date else f"{start_date} ~ {end_date}"
            if len(out_df) == 0:
                print(f"\n[A급 정형] {manu} '{mall}' 위반 상세 ({title_range})")
                print("(위반 없음)\n")
                continue
            print(f"\n[A급 정형] {manu} '{mall}' 위반 상세 ({title_range}) — {len(out_df)}행")
            print_result_any(out_df, output_dir=args.output_dir, prefix=f"A1_VIOL_DETAIL_{manu}_{mall}_{start_date}_to_{end_date}")
            print()
            continue

        if intent == "Q1":
            q_norm = normalize_range_separators(q)
            has_range_signal = any(k in q_norm for k in ["~", "-", "부터", "까지", "최근", "지난", "이번주", "지난주", "이번달", "지난달", "전월", "금주", "금월"])
            is_range = (start_date != end_date) and (has_range_signal or len(dates_list) >= 2)

            run_dates = expand_date_range_days(start_date, end_date, max_days=45) if is_range else (dates_list if dates_list else [batch_date])

            any_found = False
            for d in run_dates:
                out = q1_product_best(df_pc_all, pkey, d)
                if not out:
                    print(f"\n[정형결과] {pkey} {d} 데이터가 없습니다.")
                    continue
                any_found = True
                manu_txt = out.get("manufacturer") or catalog["key_to_manufacturer"].get(pkey, "-")
                print(f"\n[정형결과] {pkey} {d} 제조사={manu_txt} 최저가: {out['best_mall']} / {out['best_price']}원")
                print(df_to_string_kr(out["table"], index=False))
            print()
            if not any_found:
                print("데이터를 못 찾았어요.\n")
            continue

        if intent == "Q1_DETAIL":
            run_date = dates_list[0] if dates_list else batch_date
            offers = q1_product_all_offers(df_data_all, pkey, run_date)
            if offers is None:
                print(f"\n[정형결과] {pkey} {run_date} 오퍼(원본) 데이터를 못 찾았어요.\n")
                continue
            print(f"\n[정형결과] {pkey} {run_date} 오퍼 목록 {len(offers)}건")
            print_result_any(offers, output_dir=args.output_dir, prefix=f"Q1_DETAIL_{pkey}_{run_date}")
            print()
            continue

        if intent == "Q4":
            out = q4_shipping_issues(df_data_all, start_date, end_date, product_key=pkey if pkey else None)
            if out is None or len(out) == 0:
                print("데이터를 못 찾았어요.\n")
                continue
            n = parse_top_n(q, default_n=10)
            title_range = start_date if start_date == end_date else f"{start_date}~{end_date}"
            print(f"\n[정형결과] 배송 이슈 많은 몰 TOP {n} (기간 {title_range})")
            print(df_to_string_kr(out.head(n), index=False))
            print()
            continue

        if intent == "Q5":
            include_target_price = wants_target_price(q)
            include_diff = wants_diff(q)
            require_target_price = include_target_price or include_diff

            out = q5_mall_not_cheapest(
                df_pc_all,
                batch_date,
                mall,
                manufacturer=manu,
                include_target_price=include_target_price,
                include_diff=include_diff,
                require_target_price=require_target_price,
            )
            if not out:
                print("데이터를 못 찾았어요.\n")
                continue

            scope = f"(제조사={manu}) " if manu else ""
            print(
                f"\n[정형결과] {batch_date} {scope}모든 과자 중 '{out['target_mall']}'이(가) 최저가가 아닌 제품: "
                f"{out['count']}개 / 전체 {out['total_products']}개"
            )

            table: pd.DataFrame = out["table"]
            if table.empty:
                print("(해당 없음)")
            else:
                print(df_to_string_kr(table, index=False))

            ensure_dir(args.output_dir)
            mtag = manu if manu else "ALL"
            path = os.path.join(args.output_dir, f"Q5_not_cheapest_{out['target_mall']}_{mtag}_{out['batch_date']}.csv")
            table.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"\n[저장] CSV: {path}\n")
            continue

        if intent == "Q6":
            out = q6_mall_is_cheapest(df_pc_all, batch_date, mall, manufacturer=manu)
            if not out:
                print("데이터를 못 찾았어요.\n")
                continue
            scope = f"(제조사={manu}) " if manu else ""
            print(f"\n[정형결과] {batch_date} {scope}'{mall}'이(가) 최저가인 제품: {out['count']}개 / 전체 {out['total_products']}개")
            table = out["table"]
            if table.empty:
                print("(해당 없음)")
            else:
                print(df_to_string_kr(table, index=False))
            print()
            continue

        if intent == "Q7":
            n = parse_top_n(q, default_n=10)
            metric = parse_metric_kor(q)
            ascending = parse_sort_direction(q, metric)

            out = q7_mall_metric_topn(df_pc_all, batch_date, mall, metric, n, ascending, manufacturer=manu)
            if not out:
                print("데이터를 못 찾았어요(지표/날짜/몰/제조사 확인).\n")
                continue

            direction = "낮은값 TOP" if ascending else "높은값 TOP"
            scope = f"(제조사={manu}) " if manu else ""
            print(f"\n[정형결과] {batch_date} {scope}{mall} {metric} {direction} {n}개")
            print(df_to_string_kr(out["table"], index=False))
            print()
            continue

        if intent == "Q8_TREND":
            out_df = q8_product_trend(df_pc_all, product_key=pkey, start_date=start_date, end_date=end_date, manufacturer=manu, mall=mall)
            if out_df is None:
                print("기간 내 데이터가 없습니다.\n")
                continue
            mall_scope = f", 몰={mall}" if mall else ""
            manu_scope = f", 제조사={manu}" if manu else ""
            title_range = start_date if start_date == end_date else f"{start_date} ~ {end_date}"
            print(f"\n[정형결과] {pkey} 가격 추이 ({title_range}{manu_scope}{mall_scope})")
            print_result_any(out_df, output_dir=args.output_dir, prefix=f"Q8_trend_{pkey}_{start_date}_to_{end_date}")
            print()
            continue

        if intent == "Q7B_MANU_RANGE_TOPN":
            n = parse_top_n(q, default_n=20)
            metric = parse_metric_kor(q)
            ascending = parse_sort_direction(q, metric)
            out_df = q7b_manufacturer_range_topn(df_pc_all, start_date, end_date, manu, metric, n, ascending)
            if out_df is None or len(out_df) == 0:
                print("기간 내 데이터가 없습니다.\n")
                continue
            direction = "낮은값 TOP" if ascending else "높은값 TOP"
            title_range = start_date if start_date == end_date else f"{start_date} ~ {end_date}"
            print(f"\n[정형결과] {manu} {metric} {direction} {n}개 ({title_range})")
            print(df_to_string_kr(out_df, index=False))
            print()
            continue

        if intent == "Q7C_MANU_RANGE_TOPN_WITH_DATE":
            n = parse_top_n(q, default_n=20)
            metric = parse_metric_kor(q)
            ascending = parse_sort_direction(q, metric)
            out_df = q7c_manufacturer_range_topn_with_date(df_pc_all, start_date, end_date, manu, metric, n, ascending)
            if out_df is None or len(out_df) == 0:
                print("기간 내 데이터가 없습니다.\n")
                continue
            direction = "낮은값 TOP" if ascending else "높은값 TOP"
            title_range = start_date if start_date == end_date else f"{start_date} ~ {end_date}"
            print(f"\n[정형결과] {manu} {metric} {direction} {n}개 - 날짜 포함 ({title_range})")
            print(df_to_string_kr(out_df, index=False))
            print()
            continue

        if intent == "Q7D_MANU_RANGE_TOPN_BY_MALL":
            n = parse_top_n(q, default_n=5)
            metric = parse_metric_kor(q)
            ascending = parse_sort_direction(q, metric)
            out_map = q7d_manufacturer_range_topn_by_mall(df_pc_all, start_date, end_date, manu, metric, n, ascending)
            if not out_map:
                print("기간 내 데이터가 없습니다.\n")
                continue
            direction = "낮은값 TOP" if ascending else "높은값 TOP"
            title_range = start_date if start_date == end_date else f"{start_date} ~ {end_date}"
            print(f"\n[정형결과] {manu} {metric} {direction} - 몰별 TOP {n} ({title_range})")
            for mall_name in sorted(out_map.keys()):
                print(f"\n--- {mall_name} ---")
                print(df_to_string_kr(out_map[mall_name], index=False))
            print()
            continue

        if intent == "Q9_MALL_SUMMARY_TABLE":
            out_df = q9_mall_summary_table(df_pc_all, start_date, end_date, mall, manufacturer=manu)
            if out_df is None or len(out_df) == 0:
                print("기간 내 데이터가 없습니다.\n")
                continue
            title_range = start_date if start_date == end_date else f"{start_date} ~ {end_date}"
            scope = f", 제조사={manu}" if manu else ""
            print(f"\n[정형결과] {mall} 요약 가격표 ({title_range}{scope})")
            print_result_any(out_df, output_dir=args.output_dir, prefix=f"Q9_summary_{mall}_{start_date}_to_{end_date}")
            print()
            continue

        # -----------------------------
        # SAFE plan fallback
        # -----------------------------
        if should_block_plan_fallback(q):
            print("정형 규칙에 없는 위반/비교 질의입니다. 정형 라우터 보강이 필요합니다.\n")
            continue

        if not llm_plan_enabled:
            print("규칙에 없는 질문입니다. (LLM SAFE 폴백 비활성화)\n")
            continue

        print("[LLM SAFE 폴백] PandasPlan JSON plan 생성 → 실행 중...")
        try:
            entities = {"pkey": pkey, "mall": mall, "manu": manu, "batch_date": batch_date}
            llm_plan_ctx = build_llm_plan_context(
                df_pc_all, df_data_all, today, q, entities, slots, start_date, end_date
            )

            try:
                raw_plan = llm_generate_plan(
                    q,
                    llm_plan_ctx,
                    model_path=args.llm_model_path,
                    n_ctx=args.llm_n_ctx,
                    max_tokens=args.llm_max_tokens,
                    temperature=args.llm_temperature,
                    n_threads=args.llm_threads,
                    n_gpu_layers=args.llm_gpu_layers,
                )
                if args.print_plan:
                    print("\n[RAW PLAN]")
                    print(json.dumps(raw_plan, ensure_ascii=False, indent=2))
            except Exception as e:
                print(f"[WARN] LLM JSON 생성 실패 → 규칙 기반 mini-plan으로 대체: {repr(e)}")
                raw_plan = build_rule_based_miniplan(
                    q,
                    batch_date=batch_date,
                    start_date=start_date,
                    end_date=end_date,
                    mall=mall,
                    manu=manu,
                    pkey=pkey,
                )
                if args.print_plan:
                    print("\n[RULE-BASED RAW PLAN]")
                    print(json.dumps(raw_plan, ensure_ascii=False, indent=2))

            repaired_plan = repair_plan_structure(
                raw_plan,
                question=q,
                batch_date=batch_date,
                start_date=start_date,
                end_date=end_date,
                mall=mall,
                manu=manu,
                pkey=pkey,
            )
            repaired_plan = enrich_plan_semantics(
                repaired_plan,
                question=q,
                start_date=start_date,
                end_date=end_date,
                mall=mall,
                manu=manu,
                pkey=pkey,
            )
            repaired_plan = validate_plan_against_question(
                q,
                repaired_plan,
                mall=mall,
                manu=manu,
                pkey=pkey,
                start_date=start_date,
                end_date=end_date,
            )

            if args.print_plan:
                print("\n[REPAIRED RAW PLAN]")
                print(json.dumps(repaired_plan, ensure_ascii=False, indent=2))

            plan = validate_and_normalize_plan(repaired_plan, df_pc_all=df_pc_all, df_data_all=df_data_all)

            if args.print_plan:
                print_normalized_plan(plan)

            out_df = execute_plan(plan, df_pc_all=df_pc_all, df_data_all=df_data_all)

            print("\n[LLM SAFE 결과]")
            print_result_any(out_df, output_dir=args.output_dir, prefix="LLM_SAFE")
            print()
        except Exception as e:
            print("LLM SAFE 폴백 실패:", repr(e))
            print("힌트: 질문에 날짜/몰/지표/제조사(또는 자사/우리/당사)를 포함하면 성공률이 더 올라갑니다.\n")


if __name__ == "__main__":
    main()