#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import calendar
from datetime import datetime, timedelta
from typing import Any, Optional, Tuple, Dict, List

import pandas as pd

DEFAULT_OUTPUT_DIR = "./outputs"

MY_MANUFACTURER = "해태제과"
MY_WORDS = ["자사","우리","당사","우리회사","우리 회사","본사","우리제품","우리 제품","당사제품","당사 제품"]

DEFAULT_ENABLE_LLM = True
DEFAULT_PRINT_PLAN = True
DEFAULT_LLM_MODEL_PATH = "/home/siwasoft/gsllm/gemma3-27b/gemma-3-27b-it-Q4_K_M.gguf"
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
rf_process = None

COMPARE_WORDS = ["비교", "대비", "vs", "VS", "versus", "차이", "갭", "gap", "더 비싼", "더 싼", "누가 더", "어디가 더", "비교해서", "비교하면", "비교해줘", "비교해", "쪽이 더", "어느 쪽", "어느쪽"]
PRODUCT_QUERY_HINTS = ["가격", "10g", "10g당", "최저가", "최고가", "평균가", "시세", "싼", "저렴", "비싼", "비교", "추이", "상세"]
DETAIL_RAW_DEFAULT_SELECT = ["batch_date","source_sheet","source_type","mall_name","product_name","item_name","flavor","weight_g","count","price","price_per_10g","shipping_fee","seller","url","product_match_key"]
SUMMARY_DEFAULT_SELECT = ["batch_date","target_mall","compare_mall","product_name","flavor","weight_g","count","target_price_per_10g","compare_price_per_10g","diff_per_10g","cheaper_side"]

DASH_VARIANTS = ["–", "—", "−"]
WAVE_VARIANTS = ["∼", "〜"]

COLNAME_KR: Dict[str, str] = {
    "batch_date": "배치일자",
    "source_sheet": "시트명",
    "source_type": "소스구분",
    "target_mall": "기준몰",
    "compare_mall": "비교몰",
    "mall_name": "쇼핑몰",
    "seller": "대표자",
    "product_name": "제품명",
    "item_name": "품목명",
    "flavor": "맛",
    "weight_g": "용량(g)",
    "count": "개수",
    "price": "가격",
    "shipping_fee": "배송비",
    "price_per_10g": "10g당가격",
    "target_price": "기준몰가격",
    "target_price_per_10g": "쿠팡해태10g당가격",
    "compare_price": "비교몰가격",
    "compare_price_per_10g": "비교몰10g당가격",
    "diff_per_10g": "10g당차이",
    "abs_diff_per_10g": "절대10g당차이",
    "cheaper_side": "더저렴한쪽",
    "product_match_key": "제품매칭키",
    "url": "링크",
    "target_url": "기준몰링크",
    "compare_url": "비교몰링크",
}


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


def normalize_name(s: Any) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    t = s.strip()
    t = re.sub(r"해태제과\s*", "", t)
    t = re.sub(r"\b(쿠팡|로켓배송|무료배송)\b", "", t, flags=re.I)
    t = re.sub(r"\b\d+\s*(입|p|P|봉|팩|개입)\b", "", t)
    t = re.sub(r"\bx\s*1개\b", "1개", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_flavor(text: str) -> str:
    if not isinstance(text, str):
        return ""
    candidates = ["오리지널","딸기","바나나","초코","초콜릿","치즈","양파","벌꿀","허니버터","감자","고구마","사워크림","갈릭","매운맛","카라멜","코코아","화이트","녹차","짭짤","우유크림","에스프레소","쿠키","딥초코","골드","밀크","미니"]
    for c in candidates:
        if c in text:
            return c
    return ""


def extract_weight_g(text: str) -> Optional[float]:
    if not isinstance(text, str):
        return None
    m = re.search(r'(\d+(?:\.\d+)?)\s*g\b', text, flags=re.I)
    return float(m.group(1)) if m else None


def extract_count(text: str) -> Optional[int]:
    if not isinstance(text, str):
        return None
    m = re.search(r'(\d+)\s*개\b', text)
    if m:
        return int(m.group(1))
    m = re.search(r'x\s*(\d+)', text, flags=re.I)
    if m:
        return int(m.group(1))
    return 1 if text else None


def canonical_base_name(text: str) -> str:
    t = normalize_name(text)
    t = re.sub(r"\d+(?:\.\d+)?\s*g\b", "", t, flags=re.I)
    t = re.sub(r"\d+\s*개\b", "", t)
    flavor = extract_flavor(t)
    if flavor:
        t = t.replace(flavor, " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_product_match_key(product_name: str, flavor: str, weight_g: Any, count: Any) -> str:
    base = canonical_base_name(product_name)
    fl = normalize_name(flavor or "")
    wg = ""
    if weight_g is not None and not pd.isna(weight_g):
        f = float(weight_g)
        wg = str(int(f)) if f.is_integer() else str(f)
    ct = ""
    if count is not None and not pd.isna(count):
        ct = str(int(count))
    return f"{normalize_lookup_text(base)}|{normalize_lookup_text(fl)}|{wg}|{ct}"


def df_for_terminal(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return df.rename(columns=COLNAME_KR).copy()
    except Exception:
        return df.copy()


def df_to_string_kr(df: pd.DataFrame, index: bool=False) -> str:
    return df_for_terminal(df).to_string(index=index)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def now_ts() -> int:
    return int(datetime.now().timestamp())


def print_debug_json(payload: Dict[str, Any]):
    try:
        print("[DEBUG_JSON]", json.dumps(payload, ensure_ascii=False))
    except Exception:
        print("[DEBUG_JSON]", str(payload))


def print_result_any(result: Any, output_dir: str, prefix: str="RESULT"):
    if isinstance(result, pd.DataFrame):
        df = result.copy()
        if df.shape[0] >= CSV_SAVE_THRESHOLD or df.shape[0] > MAX_RESULT_ROWS_PRINT or df.shape[1] > MAX_RESULT_COLS_PRINT:
            ensure_dir(output_dir)
            path = os.path.join(output_dir, f"{prefix}_{now_ts()}.csv")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"[결과] DataFrame이 커서 CSV로 저장했습니다: {path}")
            print(df_to_string_kr(df.head(50), index=False))
            print(f"... (총 {len(df)}행, {df.shape[1]}열)")
            return
        print(df_to_string_kr(df, index=False))
        return
    if isinstance(result, (dict, list)):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result)


def normalize_batch_date_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip().str.slice(0,10)
    out = out.str.replace("/", "-", regex=False)
    return out


def today_from_df(df: pd.DataFrame) -> str:
    if "batch_date" not in df.columns or len(df)==0:
        return ""
    return str(df["batch_date"].astype(str).max())[:10]


def min_date_from_df(df: pd.DataFrame, fallback: str) -> str:
    try:
        return str(df["batch_date"].astype(str).min())[:10]
    except Exception:
        return fallback


def shift_date_ymd(date_str: str, delta_days: int) -> str:
    dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
    return (dt + timedelta(days=int(delta_days))).strftime("%Y-%m-%d")


def _month_range_for(dt: datetime) -> Tuple[str,str]:
    y,m = dt.year, dt.month
    last = calendar.monthrange(y,m)[1]
    return f"{y:04d}-{m:02d}-01", f"{y:04d}-{m:02d}-{last:02d}"


def _week_range_for(dt: datetime) -> Tuple[str,str]:
    start = dt - timedelta(days=dt.weekday())
    end = start + timedelta(days=6)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def parse_batch_date(question: str, default_date: str) -> str:
    q = normalize_range_separators(question.strip())
    if "그제" in q or "그저께" in q:
        return shift_date_ymd(default_date, -2)
    if "어제" in q:
        return shift_date_ymd(default_date, -1)
    if "오늘" in q or "금일" in q:
        return default_date
    m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", q)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return default_date


def parse_dates_list(question: str, default_date: str) -> List[str]:
    q = normalize_range_separators(question.strip())
    out=[]
    for mm in re.finditer(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", q):
        out.append(f"{int(mm.group(1)):04d}-{int(mm.group(2)):02d}-{int(mm.group(3)):02d}")
    if "어제" in q:
        out.append(shift_date_ymd(default_date, -1))
    if "오늘" in q or "금일" in q:
        out.append(default_date)
    seen=[]
    for x in out:
        if x not in seen:
            seen.append(x)
    return seen


def parse_date_range(question: str, default_end: str, default_start: str) -> Tuple[str,str]:
    q = normalize_range_separators(question.strip())
    dt_end = datetime.strptime(default_end, "%Y-%m-%d")
    if any(k in q for k in ["최근 일주일","최근 1주일","지난 일주일","지난 일주일동안","지난 일주일 동안","최근 일주일동안","최근 일주일 동안","최근 7일","지난 7일"]):
        s = (dt_end - timedelta(days=6)).strftime("%Y-%m-%d")
        return max(s, default_start), default_end
    m = re.search(r"(최근|지난)\s*([0-9]{1,3})\s*일", q)
    if m:
        n = max(1, min(int(m.group(2)),365))
        s = (dt_end - timedelta(days=n-1)).strftime("%Y-%m-%d")
        return max(s, default_start), default_end
    if any(k in q for k in ["이번주","이번 주","금주"]):
        s,e = _week_range_for(dt_end)
        return max(s, default_start), min(e, default_end)
    if any(k in q for k in ["이번달","이번 달","금월"]):
        s,e = _month_range_for(dt_end)
        return max(s, default_start), min(e, default_end)
    dl = parse_dates_list(q, default_end)
    if len(dl) >= 2:
        s,e = dl[0], dl[1]
        if s>e: s,e = e,s
        return max(s, default_start), min(e, default_end)
    d = parse_batch_date(q, default_end)
    return max(d, default_start), min(d, default_end)


def parse_requested_date_range_uncapped(question: str, default_end: str) -> Tuple[str,str]:
    q = normalize_range_separators(question.strip())
    if any(k in q for k in ["최근 일주일","최근 1주일","지난 일주일","지난 일주일동안","지난 일주일 동안","최근 일주일동안","최근 일주일 동안","최근 7일","지난 7일"]):
        e = default_end
        s = shift_date_ymd(e, -6)
        return s,e
    dl = parse_dates_list(q, default_end)
    if len(dl) >= 2:
        s,e = dl[0], dl[1]
        if s>e: s,e = e,s
        return s,e
    d = parse_batch_date(q, default_end)
    return d,d


def parse_top_n(question: str, default_n: int=10) -> int:
    q = question.strip()
    m = re.search(r"상위\s*([0-9]{1,4})\s*개", q)
    if m:
        return int(m.group(1))
    m = re.search(r"([0-9]{1,4})\s*개", q)
    if m:
        return int(m.group(1))
    return default_n


def build_catalog(df_compare: pd.DataFrame, df_data: pd.DataFrame) -> Dict[str, Any]:
    mall_series=[]
    if "compare_mall" in df_compare.columns:
        mall_series.append(df_compare["compare_mall"])
    if "mall_name" in df_data.columns:
        mall_series.append(df_data["mall_name"])
    malls = sorted(set(pd.concat(mall_series).astype(str))) if mall_series else []
    malls = [m for m in malls if m and m != "쿠팡해태"]
    manufacturers=[MY_MANUFACTURER]
    prod_frames=[]
    for df in [df_compare, df_data]:
        if len(df) and "product_match_key" in df.columns:
            cols = [c for c in ["product_name","flavor","weight_g","count","product_match_key"] if c in df.columns]
            prod_frames.append(df[cols].copy())
    if prod_frames:
        prod = pd.concat(prod_frames, ignore_index=True).drop_duplicates()
    else:
        prod = pd.DataFrame(columns=["product_name","flavor","weight_g","count","product_match_key"])
    return {"malls": malls, "manufacturers": manufacturers, "product_key_set": set(prod["product_match_key"].astype(str)) if "product_match_key" in prod.columns else set(), "products": prod.reset_index(drop=True)}


def extract_target_mall(question: str, malls: List[str]) -> Optional[str]:
    for m in sorted(malls, key=len, reverse=True):
        if m and m in question:
            return m
    return None


def extract_all_malls(question: str, malls: List[str]) -> List[str]:
    hits=[]
    for m in sorted(malls, key=len, reverse=True):
        if m and m in question and m not in hits:
            hits.append(m)
    return hits


def parse_manufacturer(question: str, manufacturers: List[str]) -> Optional[str]:
    return MY_MANUFACTURER if any(w in question for w in ["해태", "자사", "우리", "당사"]) else None


def parse_all_manufacturers(question: str, manufacturers: List[str]) -> List[str]:
    return [MY_MANUFACTURER] if parse_manufacturer(question, manufacturers) else []


def resolve_my_manufacturer(question: str, manufacturers: List[str], fallback_my: str=MY_MANUFACTURER) -> Optional[str]:
    return fallback_my if any(w in question for w in MY_WORDS) else None


def parse_product_key(question: str, catalog: Dict[str,Any]) -> Optional[str]:
    q = question.strip()
    if ("보다" in q or "쿠팡해태" in q) and not extract_weight_g(q):
        return None
    q_weight = extract_weight_g(q)
    q_count = extract_count(q)
    q_flavor = extract_flavor(q)
    candidates=[]
    products = catalog.get("products")
    if products is None or len(products)==0:
        return None
    for _, r in products.iterrows():
        name = str(r.get("product_name",""))
        base = canonical_base_name(name)
        qn = normalize_name(q)
        if base and any(tok and tok in qn for tok in base.split()[:2]):
            score = 1
            if q_flavor and str(r.get("flavor","")) == q_flavor:
                score += 2
            if q_weight is not None and pd.notna(r.get("weight_g")) and float(r["weight_g"]) == float(q_weight):
                score += 3
            if q_count is not None and pd.notna(r.get("count")) and int(r["count"]) == int(q_count):
                score += 3
            candidates.append((score, str(r.get("product_match_key",""))))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def looks_like_product_query(q: str) -> bool:
    return bool(extract_weight_g(q) or extract_count(q) or re.search(r"[가-힣A-Za-z]{2,}", q))


def fuzzy_pick(query: str, choices: List[str], cutoff: int=FUZZY_CUTOFF) -> Optional[str]:
    if not query or not choices:
        return None
    query = normalize_lookup_text(query)
    best=None
    for c in choices:
        n = normalize_lookup_text(c)
        if n in query or query in n:
            if best is None or len(c) > len(best):
                best = c
    return best


def discover_xlsx_files(folder: str, regex: str) -> List[str]:
    pat = re.compile(regex)
    return sorted([name for name in os.listdir(folder) if pat.search(name) and name.lower().endswith(".xlsx")])


def extract_date_from_filename(path: str) -> Optional[str]:
    base = os.path.basename(path)
    m = re.search(r"(\d{8})", base)
    if not m:
        return None
    raw = m.group(1)
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"


def _standardize_crawling(df: pd.DataFrame, batch_date: str, sheet_name: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["batch_date"] = batch_date
    out["source_sheet"] = sheet_name
    out["source_type"] = "crawling"
    out["mall_name"] = df.get("스토어", "").astype(str)
    out["seller"] = df.get("대표자", "").astype(str)
    out["search_keyword"] = df.get("검색키워드", "").astype(str)
    out["item_name"] = df.get("품목명", "").astype(str)
    out["product_name"] = out["search_keyword"].where(out["search_keyword"].astype(str).str.strip() != "", out["item_name"])
    out["flavor"] = out["product_name"].astype(str).apply(extract_flavor)
    out["weight_g"] = pd.to_numeric(df.get("용량(g)"), errors="coerce")
    out["count"] = pd.to_numeric(df.get("개수"), errors="coerce")
    out["price"] = pd.to_numeric(df.get("가격(원)"), errors="coerce")
    out["shipping_fee"] = pd.to_numeric(df.get("배송비(원)"), errors="coerce")
    out["price_per_10g"] = pd.to_numeric(df.get("10g당 가격"), errors="coerce")
    out["url"] = df.get("링크", "").astype(str)
    out["manufacturer"] = MY_MANUFACTURER
    out["product_match_key"] = [build_product_match_key(p, f, w, c) for p,f,w,c in zip(out["product_name"], out["flavor"], out["weight_g"], out["count"])]
    return out


def _find_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def _standardize_coupang(df: pd.DataFrame, batch_date: str, sheet_name: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["batch_date"] = batch_date
    out["source_sheet"] = sheet_name
    out["source_type"] = "coupang_haetae"
    out["mall_name"] = "쿠팡해태"
    out["seller"] = ""
    out["item_name"] = df.get("품목명", "").astype(str)
    out["product_name"] = out["item_name"]
    out["flavor"] = out["product_name"].astype(str).apply(extract_flavor)
    gcol = _find_col(df, ["g단위","용량(g)","g"])
    if gcol:
        out["weight_g"] = pd.to_numeric(df.get(gcol), errors="coerce")
    else:
        out["weight_g"] = out["product_name"].astype(str).apply(extract_weight_g)
    out["count"] = out["product_name"].astype(str).apply(extract_count)
    out["price"] = pd.to_numeric(df.get("가격"), errors="coerce")
    out["shipping_fee"] = 0
    out["price_per_10g"] = pd.to_numeric(df.get("10g당 가격"), errors="coerce")
    ucol = _find_col(df, ["링크","URL","링크코드","url"])
    out["url"] = df.get(ucol, "").astype(str) if ucol else ""
    out["manufacturer"] = MY_MANUFACTURER
    out["product_match_key"] = [build_product_match_key(p, f, w, c) for p,f,w,c in zip(out["product_name"], out["flavor"], out["weight_g"], out["count"])]
    return out


_COMPARE_COLS = ["batch_date","manufacturer","target_mall","compare_mall","product_match_key","product_name","item_name","flavor","weight_g","count","target_price","target_price_per_10g","compare_price","compare_price_per_10g","diff_per_10g","abs_diff_per_10g","cheaper_side","seller","target_url","compare_url"]


def build_compare_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    target = df_raw[df_raw["source_type"] == "coupang_haetae"].copy()
    others = df_raw[df_raw["source_type"] == "crawling"].copy()
    if len(target) == 0 or len(others) == 0:
        return pd.DataFrame(columns=_COMPARE_COLS)
    tgt = target[["batch_date","product_match_key","product_name","item_name","flavor","weight_g","count","price_per_10g","price","url"]].copy()
    tgt = tgt.rename(columns={"price_per_10g":"target_price_per_10g","price":"target_price","url":"target_url"})
    oth = others[["batch_date","product_match_key","product_name","item_name","flavor","weight_g","count","mall_name","seller","price_per_10g","price","url"]].copy()
    oth = oth.rename(columns={"mall_name":"compare_mall","price_per_10g":"compare_price_per_10g","price":"compare_price","url":"compare_url"})
    merged = oth.merge(tgt, on=["batch_date","product_match_key"], how="inner", suffixes=("_cmp","_tgt"))
    if len(merged) == 0:
        return pd.DataFrame(columns=_COMPARE_COLS)
    merged["product_name"] = merged["product_name_tgt"].fillna(merged["product_name_cmp"])
    merged["item_name"] = merged["item_name_cmp"].fillna(merged["item_name_tgt"])
    merged["flavor"] = merged["flavor_tgt"].fillna(merged["flavor_cmp"])
    merged["weight_g"] = merged["weight_g_tgt"].fillna(merged["weight_g_cmp"])
    merged["count"] = merged["count_tgt"].fillna(merged["count_cmp"])
    merged["target_mall"] = "쿠팡해태"
    merged["manufacturer"] = MY_MANUFACTURER
    merged["diff_per_10g"] = pd.to_numeric(merged["compare_price_per_10g"], errors="coerce") - pd.to_numeric(merged["target_price_per_10g"], errors="coerce")
    merged["abs_diff_per_10g"] = merged["diff_per_10g"].abs()
    merged["cheaper_side"] = merged["diff_per_10g"].apply(lambda x: "쿠팡해태" if pd.notna(x) and x > 0 else ("비교몰" if pd.notna(x) and x < 0 else "동일"))
    out = merged[_COMPARE_COLS].copy()
    return out.sort_values(["batch_date","product_match_key","compare_mall","compare_price_per_10g"], ascending=[True,True,True,True]).reset_index(drop=True)


def load_excel_one(xlsx_path: str) -> Tuple[pd.DataFrame,pd.DataFrame]:
    batch_date = extract_date_from_filename(xlsx_path)
    if batch_date is None:
        raise ValueError(f"파일명에서 날짜를 찾을 수 없습니다: {xlsx_path}")
    xl = pd.ExcelFile(xlsx_path)
    crawl_sheet = next((s for s in xl.sheet_names if s.startswith("크롤링_")), None)
    coupang_sheet = next((s for s in xl.sheet_names if s.startswith("쿠팡해태_")), None)
    if crawl_sheet is None or coupang_sheet is None:
        raise ValueError("필수 시트(크롤링_*, 쿠팡해태_*)를 찾지 못했습니다.")
    df_crawl = pd.read_excel(xlsx_path, sheet_name=crawl_sheet)
    df_cp = pd.read_excel(xlsx_path, sheet_name=coupang_sheet)
    raw_crawl = _standardize_crawling(df_crawl, batch_date, crawl_sheet)
    raw_cp = _standardize_coupang(df_cp, batch_date, coupang_sheet)
    df_data = pd.concat([raw_crawl, raw_cp], ignore_index=True)
    df_compare = build_compare_from_raw(df_data)
    return df_data, df_compare


def load_excels_multi(folder: str, filenames: List[str]) -> Tuple[pd.DataFrame,pd.DataFrame]:
    data_list=[]
    comp_list=[]
    for fn in filenames:
        dfd, dfc = load_excel_one(os.path.join(folder, fn))
        data_list.append(dfd)
        comp_list.append(dfc)
    df_data = pd.concat(data_list, ignore_index=True) if data_list else pd.DataFrame()
    df_compare = pd.concat(comp_list, ignore_index=True) if comp_list else pd.DataFrame(columns=_COMPARE_COLS)
    if len(df_compare) == 0:
        for c in _COMPARE_COLS:
            if c not in df_compare.columns:
                df_compare[c] = pd.Series(dtype="object")
    return df_data, df_compare
