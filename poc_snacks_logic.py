#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from typing import Any, Dict, Optional

import pandas as pd

COMPARE_WORDS = ["비교", "대비", "vs", "차이", "갭", "보다", "비싼", "싼", "저렴", "높은", "낮은"]
PRICE_WORDS = ["가격", "10g", "10g당", "비싼", "싼", "저렴", "높은", "낮은"]
DETAIL_WORDS = ["상세", "원본", "링크", "목록", "리스트", "전부", "전체", "모두"]
TREND_WORDS = ["추이", "트렌드", "변화", "기간", "최근", "지난", "일별", "날짜별", "동안"]


def wants_detail(q: str) -> bool:
    return any(k in q for k in DETAIL_WORDS)


def wants_trend(q: str) -> bool:
    return any(k in q for k in TREND_WORDS)


def wants_compare(q: str) -> bool:
    return any(k in q for k in COMPARE_WORDS)


def wants_price(q: str) -> bool:
    return any(k in q for k in PRICE_WORDS)


def extract_semantic_slots(q: str) -> Dict[str, Any]:
    return {
        "wants_detail": wants_detail(q),
        "wants_trend": wants_trend(q),
        "wants_compare": wants_compare(q),
        "wants_price": wants_price(q),
        "has_range": any(k in q for k in ["최근", "지난", "~", "부터", "까지", "동안"]),
        "view_source": "compare" if wants_compare(q) or wants_price(q) else "raw",
    }


def infer_time_mode(q: str, start_date: str, end_date: str, slots: Dict[str, Any]) -> str:
    if slots.get("wants_trend"):
        return "trend"
    if start_date != end_date:
        return "range"
    return "single"


def _ensure_compare_columns(cur: pd.DataFrame) -> pd.DataFrame:
    cur = cur.copy()
    for c in ["batch_date","manufacturer","target_mall","compare_mall","product_match_key","product_name","item_name","flavor","weight_g","count","target_price","target_price_per_10g","compare_price","compare_price_per_10g","diff_per_10g","abs_diff_per_10g","cheaper_side","seller","target_url","compare_url"]:
        if c not in cur.columns:
            cur[c] = pd.NA
    cur["diff_per_10g"] = pd.to_numeric(cur["diff_per_10g"], errors="coerce")
    cur["abs_diff_per_10g"] = pd.to_numeric(cur["abs_diff_per_10g"], errors="coerce")
    cur["abs_diff_per_10g"] = cur["abs_diff_per_10g"].fillna(cur["diff_per_10g"].abs())
    return cur


def parse_compare_direction(q: str, mall: Optional[str]) -> str:
    has_low = any(w in q for w in ["싼", "저렴", "낮은"])
    has_high = any(w in q for w in ["비싼", "높은"])
    if mall and (f"{mall}보다" in q or mall in q):
        if has_high:
            return "target_more_expensive_than_compare"
        if has_low:
            return "target_cheaper_than_compare"
    if "쿠팡해태보다" in q:
        if has_high:
            return "compare_more_expensive_than_target"
        if has_low:
            return "compare_cheaper_than_target"
    if has_high:
        return "target_more_expensive_than_compare"
    if has_low:
        return "target_cheaper_than_compare"
    return "all"


def compare_with_mall(df_compare_all: pd.DataFrame, *, start_date: str, end_date: str, compare_mall: str, direction: str, product_key: Optional[str]=None, summary_only: bool=False) -> pd.DataFrame:
    cur = _ensure_compare_columns(df_compare_all)
    cur = cur[(cur["batch_date"] >= start_date) & (cur["batch_date"] <= end_date)].copy()
    if compare_mall:
        cur = cur[cur["compare_mall"].astype(str) == str(compare_mall)].copy()
    if product_key:
        cur = cur[cur["product_match_key"].astype(str) == str(product_key)].copy()
    if direction in ["target_more_expensive_than_compare", "compare_cheaper_than_target"]:
        cur = cur[cur["diff_per_10g"] < 0].copy()
    elif direction in ["target_cheaper_than_compare", "compare_more_expensive_than_target"]:
        cur = cur[cur["diff_per_10g"] > 0].copy()
    cur = cur.sort_values(["batch_date","abs_diff_per_10g","compare_mall","product_match_key"], ascending=[True,False,True,True]).reset_index(drop=True)
    cols = ["batch_date","manufacturer","target_mall","compare_mall","product_match_key","product_name","flavor","weight_g","count","target_price","target_price_per_10g","compare_price","compare_price_per_10g","diff_per_10g","abs_diff_per_10g","cheaper_side","seller","target_url","compare_url"]
    keep = [c for c in cols if c in cur.columns]
    return cur[keep].reset_index(drop=True)


def product_compare(df_compare_all: pd.DataFrame, *, product_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    return compare_with_mall(df_compare_all, start_date=start_date, end_date=end_date, compare_mall="", direction="all", product_key=product_key, summary_only=False)


def q8_product_trend(df_pc: pd.DataFrame, product_key: str, start_date: str, end_date: str, manufacturer: Optional[str]=None, mall: Optional[str]=None) -> Optional[pd.DataFrame]:
    cur = _ensure_compare_columns(df_pc)
    cur = cur[(cur["product_match_key"] == product_key) & (cur["batch_date"] >= start_date) & (cur["batch_date"] <= end_date)].copy()
    if mall:
        cur = cur[cur["compare_mall"] == mall].copy()
    if len(cur) == 0:
        return None
    cols = ["batch_date","compare_mall","product_match_key","product_name","flavor","weight_g","count","target_price_per_10g","compare_price_per_10g","diff_per_10g","target_url","compare_url"]
    keep = [c for c in cols if c in cur.columns]
    return cur[keep].sort_values(["batch_date","compare_mall"], ascending=[True,True]).reset_index(drop=True)


def q1_product_all_offers(df_data_all: pd.DataFrame, product_key: str, batch_date: str) -> Optional[pd.DataFrame]:
    cur = df_data_all.copy()
    cur = cur[(cur["product_match_key"] == product_key) & (cur["batch_date"] == batch_date)].copy()
    if len(cur) == 0:
        return None
    cols = [c for c in ["batch_date","source_sheet","source_type","mall_name","seller","product_name","item_name","flavor","weight_g","count","price","shipping_fee","price_per_10g","url","product_match_key"] if c in cur.columns]
    return cur[cols].sort_values(["source_type","mall_name"], ascending=[True,True]).reset_index(drop=True)


def q1_product_best(df_pc: pd.DataFrame, product_key: str, batch_date: str):
    out = product_compare(df_pc, product_key=product_key, start_date=batch_date, end_date=batch_date)
    if len(out) == 0:
        return None
    best_abs = pd.to_numeric(out["abs_diff_per_10g"], errors="coerce").min()
    best_rows = out[pd.to_numeric(out["abs_diff_per_10g"], errors="coerce") == best_abs].copy()
    best_rows = best_rows.sort_values(["compare_mall","product_match_key"], ascending=[True,True]).reset_index(drop=True)
    return {"manufacturer": str(best_rows.iloc[0].get("manufacturer","")), "best_mall": ", ".join(best_rows["compare_mall"].astype(str).tolist()), "best_price": float(best_rows.iloc[0].get("compare_price_per_10g",0) or 0), "best_malls_count": int(len(best_rows)), "table": out}


def detect_intent(q: str, *, has_product: bool, has_mall: bool, start_date: str, end_date: str) -> str:
    slots = extract_semantic_slots(q)
    if has_mall and (slots["wants_compare"] or slots["wants_price"]):
        return "COMPARE_WITH_MALL"
    if has_product and slots["wants_trend"]:
        return "Q8_TREND"
    if has_product and slots["wants_compare"]:
        return "PRODUCT_COMPARE"
    if has_product and slots["wants_detail"]:
        return "PRODUCT_RAW"
    return "UNKNOWN"


LLM = None
LLM_MODEL_PATH_CACHED = None


def init_local_llm(model_path: str, n_ctx: int, n_threads: int, n_gpu_layers: int):
    global LLM, LLM_MODEL_PATH_CACHED
    if LLM is not None and LLM_MODEL_PATH_CACHED == model_path:
        return LLM
    from llama_cpp import Llama  # type: ignore
    LLM = Llama(model_path=model_path, n_ctx=int(n_ctx), n_threads=int(n_threads), n_gpu_layers=int(n_gpu_layers), verbose=False)
    LLM_MODEL_PATH_CACHED = model_path
    return LLM


def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def llm_generate_query_spec(question: str, *, today: str="", start_date: str="", end_date: str="", malls: list=None, mall_candidates: list=None, has_product: bool=False, model_path: str="", n_ctx: int=16384, max_tokens: int=160, temperature: float=0.2, n_threads: int=8, n_gpu_layers: int=0) -> Dict[str, Any]:
    malls = mall_candidates if mall_candidates is not None else (malls or [])
    default = {"intent":"UNKNOWN","direction":"all","need_summary":False,"need_raw":False,"compare_mall":None}
    try:
        llm = init_local_llm(model_path, n_ctx, n_threads, n_gpu_layers)
        prompt = (
            "한국어 질문을 compare/raw 조회용 JSON으로 바꾸세요. 설명 없이 JSON만 출력하세요.\n"
            "schema={\"intent\":\"COMPARE_WITH_MALL|PRODUCT_COMPARE|Q8_TREND|PRODUCT_RAW|UNKNOWN\",\"direction\":\"target_more_expensive_than_compare|target_cheaper_than_compare|all\",\"need_summary\":true|false,\"need_raw\":true|false,\"compare_mall\":string|null}\n"
            f"today={today}, start_date={start_date}, end_date={end_date}\n"
            f"mall candidates={malls[:100]}\n"
            f"has_product={has_product}\n"
            f"question={question}\n"
        )
        out = llm(prompt, max_tokens=int(min(160, max_tokens)), temperature=float(min(0.2, temperature)), top_p=0.9)
        obj = _extract_json_obj(str(out["choices"][0]["text"]))
        if isinstance(obj, dict):
            res = default.copy(); res.update(obj); return res
    except Exception:
        pass
    return default


def llm_generate_compare_spec(question: str, **kwargs):
    return llm_generate_query_spec(question, **kwargs)
