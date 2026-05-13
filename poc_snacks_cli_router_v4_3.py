#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import re
import json
import argparse
from typing import Any, Dict, Optional

import pandas as pd

from poc_snacks_shared import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLL_REVIEWS,
    DEFAULT_COLL_SUMMARY,
    DEFAULT_ENABLE_LLM,
    DEFAULT_LLM_GPU_LAYERS,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL_PATH,
    DEFAULT_LLM_N_CTX,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_THREADS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PRINT_PLAN,
    FUZZY_CUTOFF,
    HARD_LIMIT_ROWS,
    MY_MANUFACTURER,
    PRODUCT_QUERY_HINTS,
    build_catalog,
    df_to_string_kr,
    discover_xlsx_files,
    ensure_dir,
    expand_date_range_days,
    extract_all_malls,
    extract_target_mall,
    fuzzy_pick,
    get_chroma_collections,
    load_excels_multi,
    looks_like_product_query,
    min_date_from_df,
    normalize_batch_date_series,
    normalize_range_separators,
    parse_all_manufacturers,
    parse_batch_date,
    parse_date_range,
    parse_dates_list,
    parse_manufacturer,
    parse_product_key,
    parse_requested_date_range_uncapped,
    parse_top_n,
    print_debug_json,
    print_result_any,
    resolve_my_manufacturer,
    rf_process,
    today_from_df,
    upsert_price_compare,
    upsert_reviews_digest,
)
from poc_snacks_logic import (
    a1_violation_detail,
    a1_violation_trend,
    build_llm_plan_context,
    build_rule_based_miniplan,
    detect_intent,
    enrich_plan_semantics,
    execute_plan,
    extract_semantic_slots,
    infer_time_mode,
    llm_classify_intent,
    llm_generate_plan,
    parse_metric_kor,
    parse_sort_direction,
    print_normalized_plan,
    q1_product_all_offers,
    q1_product_best,
    q4_shipping_issues,
    q5_mall_not_cheapest,
    q5_range_detail,
    q5_range_summary,
    q6_mall_is_cheapest,
    q7_mall_metric_topn,
    q7b_manufacturer_range_topn,
    q7c_manufacturer_range_topn_with_date,
    q7d_manufacturer_range_topn_by_mall,
    q8_product_trend,
    q9_mall_summary_table,
    repair_plan_structure,
    should_block_plan_fallback,
    validate_and_normalize_plan,
    validate_plan_against_question,
    wants_all_rows,
    wants_diff,
    wants_target_price,
)


# ============================================================
# Follow-up memory helpers
# ============================================================
FOLLOWUP_PRODUCT_SUMMARY_WORDS = [
    "제품별로 요약", "제품별 요약", "제품별로 정리", "제품 단위로 요약",
    "상품별로 요약", "상품별 요약", "품목별 요약"
]
FOLLOWUP_DIFF_DESC_WORDS = [
    "차액 큰 순", "차액이 큰 순", "차액 큰순", "차액순",
    "차이 큰 순", "차이가 큰 순", "큰 순으로 정리", "차액 기준으로 정리"
]
FOLLOWUP_REFER_WORDS = [
    "위의 결과", "방금 결과", "이 결과", "직전 결과", "위 결과", "앞의 결과", "이 표", "위 표"
]


def _is_followup_query(q: str) -> bool:
    return any(k in q for k in FOLLOWUP_REFER_WORDS) or any(k in q for k in FOLLOWUP_PRODUCT_SUMMARY_WORDS + FOLLOWUP_DIFF_DESC_WORDS)


def _followup_product_summary(last_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    required = {"product_key", "product_name"}
    if not required.issubset(set(last_df.columns)):
        return None

    df = last_df.copy()

    group_cols = []
    for c in ["batch_date", "product_key", "product_name"]:
        if c in df.columns:
            group_cols.append(c)

    agg_map: Dict[str, Any] = {}
    rename_map: Dict[str, str] = {}

    if "target_mall_price" in df.columns:
        agg_map["target_mall_price"] = "max"
        rename_map["target_mall_price"] = "target_mall_price"
    if "cheaper_price" in df.columns:
        agg_map["cheaper_price"] = ["min", "max"]
    if "diff" in df.columns:
        agg_map["diff"] = ["max", "sum"]
    if "cheaper_mall" in df.columns:
        agg_map["cheaper_mall"] = "nunique"

    if not agg_map:
        return None

    out = df.groupby(group_cols, dropna=False).agg(agg_map)
    out.columns = [
        "_".join([x for x in col if x]).strip("_") if isinstance(col, tuple) else str(col)
        for col in out.columns.to_flat_index()
    ]
    out = out.reset_index()

    rename_cols = {}
    if "cheaper_price_min" in out.columns:
        rename_cols["cheaper_price_min"] = "min_cheaper_price"
    if "cheaper_price_max" in out.columns:
        rename_cols["cheaper_price_max"] = "max_cheaper_price"
    if "diff_max" in out.columns:
        rename_cols["diff_max"] = "max_diff"
    if "diff_sum" in out.columns:
        rename_cols["diff_sum"] = "sum_diff"
    if "cheaper_mall_nunique" in out.columns:
        rename_cols["cheaper_mall_nunique"] = "cheaper_mall_count"
    out = out.rename(columns=rename_cols)

    if "product_key_count" in out.columns:
        out = out.rename(columns={"product_key_count": "case_count"})

    case_count = df.groupby(group_cols, dropna=False).size().reset_index(name="case_count")
    out = out.merge(case_count, on=group_cols, how="left")

    order_cols = []
    for c in [
        "batch_date", "product_key", "product_name", "target_mall_price",
        "min_cheaper_price", "max_cheaper_price", "max_diff", "sum_diff",
        "cheaper_mall_count", "case_count"
    ]:
        if c in out.columns:
            order_cols.append(c)
    out = out[order_cols]

    sort_cols = []
    ascending = []
    if "max_diff" in out.columns:
        sort_cols.append("max_diff")
        ascending.append(False)
    if "sum_diff" in out.columns:
        sort_cols.append("sum_diff")
        ascending.append(False)
    if "product_key" in out.columns:
        sort_cols.append("product_key")
        ascending.append(True)

    if sort_cols:
        out = out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    return out


def _followup_sort_diff_desc(last_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if "diff" not in last_df.columns:
        return None

    sort_cols = ["diff"]
    ascending = [False]

    if "batch_date" in last_df.columns:
        sort_cols.append("batch_date")
        ascending.append(True)
    if "product_key" in last_df.columns:
        sort_cols.append("product_key")
        ascending.append(True)
    if "cheaper_price" in last_df.columns:
        sort_cols.append("cheaper_price")
        ascending.append(True)

    return last_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def try_handle_followup(q: str, last_result_df: Optional[pd.DataFrame], output_dir: str) -> bool:
    if last_result_df is None or len(last_result_df) == 0:
        return False
    if not _is_followup_query(q):
        return False

    out_df: Optional[pd.DataFrame] = None
    title = ""

    if any(k in q for k in FOLLOWUP_PRODUCT_SUMMARY_WORDS):
        out_df = _followup_product_summary(last_result_df)
        title = "[후속정리] 직전 결과 제품별 요약"
    elif any(k in q for k in FOLLOWUP_DIFF_DESC_WORDS):
        out_df = _followup_sort_diff_desc(last_result_df)
        title = "[후속정리] 직전 결과 차액 큰 순 정렬"

    if out_df is None:
        print("직전 결과를 요청하신 형태로 다시 정리하기 어렵습니다.\n")
        return True

    print("\n아래 표는 직전 결과를 다시 정리한 것입니다.\n")
    print(title)
    print_result_any(out_df, output_dir=output_dir, prefix="FOLLOWUP_LAST_RESULT")
    print("\n원하시면 이 결과를 다시 제품별, 몰별, 차액 기준으로도 정리해드릴 수 있어요.\n")
    return True


# ============================================================
# Friendly wrap
# ============================================================
def print_friendly_prefix():
    print("\n결과를 정리했어요. 아래 표는 계산된 내용을 그대로 보여줍니다.\n")


def print_friendly_suffix():
    print("\n필요하시면 이 결과를 제품별 요약이나 차액 큰 순으로도 다시 정리해드릴 수 있어요.\n")


# ============================================================
# Noise guard
# ============================================================
def should_reject_underspecified_unknown_question(
    q: str,
    *,
    pkey: Optional[str],
    mall: Optional[str],
    manu: Optional[str],
    slots: Dict[str, Any],
) -> bool:
    if pkey or mall or manu:
        return False

    if slots.get("has_metric"):
        return False
    if slots.get("wants_price"):
        return False
    if slots.get("wants_detail"):
        return False
    if slots.get("wants_trend"):
        return False
    if slots.get("has_compare_words"):
        return False
    if slots.get("has_violation"):
        return False
    if slots.get("wants_shipping_issue"):
        return False
    if slots.get("prefers_summary"):
        return False
    if slots.get("prefers_raw"):
        return False

    text = re.sub(r"\s+", "", q.strip())
    if len(text) <= 10:
        return True
    return False


# ============================================================
# Argparser
# ============================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Snacks POC CLI Router v4.3.0 (split, gemma3)")
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


# ============================================================
# Main
# ============================================================
def main():
    args = build_argparser().parse_args()

    print("=== Snacks POC CLI Router v4.3.0 (split, gemma3) ===")
    print(f"(자사 제조사 고정) MY_MANUFACTURER = {MY_MANUFACTURER}")
    print(f"[DEFAULT LLM MODEL] {args.llm_model_path}")
    print(f"[DEFAULT LLM CTX] n_ctx={args.llm_n_ctx}, gpu_layers={args.llm_gpu_layers}, print_plan={args.print_plan}, enable_llm={args.enable_llm}")

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
            upsert_price_compare(col_sum, df_pc_all, version="v4_3_0")
            if not args.disable_reviews_digest:
                upsert_reviews_digest(col_rev, df_data_all, version="v4_3_0")
            print("Chroma 업서트 완료")
        except Exception as e:
            print("Chroma 오류(무시하고 계속):", repr(e))

    today = today_from_df(df_pc_all)
    default_start = min_date_from_df(df_pc_all, fallback=today)
    print(f"\n현재 합본 기준 today(batch_date) = {today} (default_start={default_start})\n")

    enable_llm = bool(args.enable_llm)
    llm_plan_enabled = enable_llm and (not args.disable_llm_plan)

    if enable_llm:
        print("[LLM] intent-only 폴백 활성화 (UNKNOWN일 때 intent 분류 시도)")
        if llm_plan_enabled:
            print("[LLM] PandasPlan SAFE 폴백도 활성화")
        else:
            print("[LLM] PandasPlan SAFE 폴백 비활성화(intent-only만)")
        print("[LLM] 정형 결과 앞뒤 자연어 wrap 활성화\n")
    else:
        print("[LLM] 비활성화\n")

    print("질문 입력. 종료: exit\n")

    last_result_df: Optional[pd.DataFrame] = None

    while True:
        q_raw = input("Q> ").strip()
        if q_raw.lower() in ["exit", "quit"]:
            break
        if not q_raw:
            continue

        q = normalize_range_separators(q_raw)

        if try_handle_followup(q, last_result_df=last_result_df, output_dir=args.output_dir):
            continue

        pkey = parse_product_key(q, catalog)
        mall = extract_target_mall(q, catalog["malls"])
        manu = parse_manufacturer(q, catalog["manufacturers"])
        mall_list = extract_all_malls(q, catalog["malls"])
        manu_list = parse_all_manufacturers(q, catalog["manufacturers"])

        if mall is None:
            mall = fuzzy_pick(q, catalog["malls"], cutoff=FUZZY_CUTOFF)
        if manu is None:
            manu = fuzzy_pick(q, catalog["manufacturers"], cutoff=FUZZY_CUTOFF)

        if mall and mall not in mall_list:
            mall_list.append(mall)
        if manu and manu not in manu_list:
            manu_list.append(manu)

        my_manu = resolve_my_manufacturer(q, catalog["manufacturers"], fallback_my=MY_MANUFACTURER)
        if not manu and my_manu:
            manu = my_manu
        if my_manu and my_manu not in manu_list:
            manu_list.append(my_manu)

        dates_list = parse_dates_list(q, default_date=today)
        batch_date = dates_list[0] if dates_list else parse_batch_date(q, default_date=today)
        start_date, end_date = parse_date_range(q, default_end=today, default_start=default_start)
        requested_start_uncapped, requested_end_uncapped = parse_requested_date_range_uncapped(q, default_end=today)

        slots = extract_semantic_slots(q)
        time_mode = infer_time_mode(q, start_date, end_date, slots)
        intent = detect_intent(
            q,
            has_product=bool(pkey),
            has_mall=bool(mall),
            has_manu=bool(manu),
            multi_mall=len(mall_list) >= 2,
            multi_manu=len(manu_list) >= 2,
            start_date=start_date,
            end_date=end_date,
        )
        plan_allowed = not should_block_plan_fallback(q, slots)

        entities_for_debug = {
            "pkey": {"value": pkey, "method": "detected" if pkey else "none", "matched": pkey, "score": 100.0 if pkey else None},
            "mall": {"value": mall, "method": "detected" if mall else "none", "matched": mall, "score": 100.0 if mall else None},
            "manu": {"value": manu, "method": "detected" if manu else "none", "matched": manu, "score": 100.0 if manu else None},
            "batch_date": batch_date,
            "start_date": start_date,
            "end_date": end_date,
            "requested_start_uncapped": requested_start_uncapped,
            "requested_end_uncapped": requested_end_uncapped,
        }
        print_debug_json({
            "event": "parsed_question",
            "q": q,
            "entities": entities_for_debug,
            "slots": slots,
            "view_source": slots.get("view_source", "unknown"),
            "time_mode": time_mode,
            "intent_before_llm": intent,
            "plan_allowed": plan_allowed,
        })

        if intent == "UNKNOWN" and should_reject_underspecified_unknown_question(
            q,
            pkey=pkey,
            mall=mall,
            manu=manu,
            slots=slots,
        ):
            print("질문 의미를 더 구체적으로 적어주세요. 날짜, 쇼핑몰, 제조사, 제품, 가격/비교 조건 중 하나 이상을 포함하면 더 정확히 찾을 수 있어요.\n")
            continue

        if enable_llm and intent == "UNKNOWN":
            try:
                intent2 = llm_classify_intent(
                    q,
                    today=today,
                    default_start=default_start,
                    model_path=args.llm_model_path,
                    n_ctx=args.llm_n_ctx,
                    max_tokens=min(240, args.llm_max_tokens),
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
            print(
                f"[ROUTE] intent={intent} pkey={pkey} mall={mall} manu={manu} "
                f"batch_date={batch_date} start_date={start_date} end_date={end_date}"
            )

        if intent in ["A1_VIOL_TREND", "A1_VIOL_DETAIL"] and not manu:
            manu = MY_MANUFACTURER

        if intent == "A1_VIOL_TREND":
            if requested_end_uncapped > today:
                print(f"[안내] 종료일 {requested_end_uncapped}은 데이터 범위를 벗어나 {today}까지로 계산했습니다.")
            out_df = a1_violation_trend(df_pc_all, start_date, end_date, manu, mall)
            if out_df is None or len(out_df) == 0:
                print("기간 내 데이터가 없습니다.\n")
                continue
            title_range = start_date if start_date == end_date else f"{start_date} ~ {end_date}"
            print_friendly_prefix()
            print(f"[A급 정형] {manu} '{mall}' 위반 추이 ({title_range})")
            print_result_any(out_df, output_dir=args.output_dir, prefix=f"A1_VIOL_TREND_{manu}_{mall}_{start_date}_to_{end_date}")
            print_friendly_suffix()
            last_result_df = out_df.copy()
            continue

        if intent == "A1_VIOL_DETAIL":
            if requested_end_uncapped > today:
                print(f"[안내] 종료일 {requested_end_uncapped}은 데이터 범위를 벗어나 {today}까지로 계산했습니다.")
            out_df = a1_violation_detail(df_pc_all, start_date, end_date, manu, mall, limit=5000)
            if out_df is None:
                print("기간 내 데이터가 없습니다.\n")
                continue
            title_range = start_date if start_date == end_date else f"{start_date} ~ {end_date}"
            if len(out_df) == 0:
                print_friendly_prefix()
                print(f"[A급 정형] {manu} '{mall}' 위반 상세 ({title_range})")
                print("(위반 없음)\n")
                continue
            product_count = int(out_df["product_key"].nunique()) if "product_key" in out_df.columns else 0
            case_count = int(len(out_df))
            print_friendly_prefix()
            print(f"[A급 정형] {manu} '{mall}' 위반 상세 ({title_range}) — 제품 {product_count}개 / 케이스 {case_count}행")
            print_result_any(out_df, output_dir=args.output_dir, prefix=f"A1_VIOL_DETAIL_{manu}_{mall}_{start_date}_to_{end_date}")
            print_friendly_suffix()
            last_result_df = out_df.copy()
            continue

        if intent == "Q1":
            if not pkey:
                print("제품을 인식 못했어요. (엑셀에 존재하는 제품코드/제품명이 질문에 포함되어야 함)\n")
                continue

            q_norm = normalize_range_separators(q)
            has_range_signal = (
                "~" in q_norm
                or ("부터" in q_norm and "까지" in q_norm)
                or any(k in q_norm for k in ["최근", "지난", "이번주", "이번 주", "지난주", "지난 주", "이번달", "이번 달", "지난달", "지난 달", "전월", "금주", "금월"])
                or bool(re.search(r"\d{1,2}[-/]\d{1,2}\s*~\s*\d{1,2}[-/]\d{1,2}", q_norm))
                or bool(re.search(r"\d{4}-\d{2}-\d{2}\s*~\s*\d{4}-\d{2}-\d{2}", q_norm))
            )
            is_range = (start_date != end_date) and (has_range_signal or len(dates_list) >= 2)

            run_dates = expand_date_range_days(start_date, end_date, max_days=45) if is_range else (dates_list if dates_list else [batch_date])

            any_found = False
            last_table = None
            print_friendly_prefix()
            for d in run_dates:
                out = q1_product_best(df_pc_all, pkey, d)
                if not out:
                    print(f"\n[정형결과] {pkey} {d} 데이터가 없습니다.")
                    continue
                any_found = True
                manu_txt = out.get("manufacturer") or catalog["key_to_manufacturer"].get(pkey, "-")
                multi_note = f" (공동최저가 {out['best_malls_count']}곳)" if int(out.get("best_malls_count", 1)) > 1 else ""
                print(f"\n[정형결과] {pkey} {d} 제조사={manu_txt} 최저가: {out['best_mall']} / {out['best_price']}원{multi_note}")
                print(df_to_string_kr(out["table"], index=False))
                last_table = out["table"].copy()
            print()
            if not any_found:
                print("데이터를 못 찾았어요.\n")
            else:
                print_friendly_suffix()
                if last_table is not None:
                    last_result_df = last_table.copy()
            continue

        if intent == "Q1_DETAIL":
            if not pkey:
                print("제품을 인식 못했어요. (엑셀에 존재하는 제품코드/제품명이 질문에 포함되어야 함)\n")
                continue
            run_date = dates_list[0] if dates_list else batch_date
            offers = q1_product_all_offers(df_data_all, pkey, run_date)
            if offers is None:
                print(f"\n[정형결과] {pkey} {run_date} 오퍼(원본) 데이터를 못 찾았어요.\n")
                continue
            print_friendly_prefix()
            print(f"[정형결과] {pkey} {run_date} 오퍼 목록 {len(offers)}건")
            print_result_any(offers, output_dir=args.output_dir, prefix=f"Q1_DETAIL_{pkey}_{run_date}")
            print_friendly_suffix()
            last_result_df = offers.copy()
            continue

        if intent == "Q4":
            out = q4_shipping_issues(df_data_all, start_date, end_date, product_key=pkey if pkey else None)
            if out is None or len(out) == 0:
                print("데이터를 못 찾았어요.\n")
                continue
            n = parse_top_n(q, default_n=10)
            title_range = start_date if start_date == end_date else f"{start_date}~{end_date}"
            out_head = out.head(n).copy()
            print_friendly_prefix()
            print(f"[정형결과] 배송 이슈 많은 몰 TOP {n} (기간 {title_range})")
            print(df_to_string_kr(out_head, index=False))
            print_friendly_suffix()
            last_result_df = out_head.copy()
            continue

        if intent == "Q5_RANGE_SUMMARY":
            if not mall:
                print("쇼핑몰명을 인식 못했어요. (엑셀에 존재하는 mall_name을 질문에 포함)\n")
                continue
            if requested_end_uncapped > today:
                print(f"[안내] 종료일 {requested_end_uncapped}은 데이터 범위를 벗어나 {today}까지로 계산했습니다.")
            out_df = q5_range_summary(df_pc_all, start_date, end_date, mall, manufacturer=manu)
            if out_df is None or len(out_df) == 0:
                print("기간 내 데이터가 없습니다.\n")
                continue
            scope = f", 제조사={manu}" if manu else ""
            print_friendly_prefix()
            print(f"[정형결과] {mall}보다 싼 몰별 위반건수 (날짜별) ({start_date} ~ {end_date}{scope})")
            print_result_any(out_df, output_dir=args.output_dir, prefix=f"Q5_RANGE_SUMMARY_{mall}_{start_date}_to_{end_date}")
            print_friendly_suffix()
            last_result_df = out_df.copy()
            continue

        if intent == "Q5_RANGE_DETAIL":
            if not mall:
                print("쇼핑몰명을 인식 못했어요. (엑셀에 존재하는 mall_name을 질문에 포함)\n")
                continue
            if requested_end_uncapped > today:
                print(f"[안내] 종료일 {requested_end_uncapped}은 데이터 범위를 벗어나 {today}까지로 계산했습니다.")
            out_df = q5_range_detail(
                df_pc_all,
                start_date,
                end_date,
                mall,
                manufacturer=manu,
                include_target_price=True,
                include_diff=True,
                limit=HARD_LIMIT_ROWS if wants_all_rows(q) else 5000,
            )
            if out_df is None or len(out_df) == 0:
                print("기간 내 데이터가 없습니다.\n")
                continue
            scope = f", 제조사={manu}" if manu else ""
            product_count = int(out_df["product_key"].nunique()) if "product_key" in out_df.columns else 0
            case_count = int(len(out_df))
            print_friendly_prefix()
            print(f"[정형결과] {mall}보다 싼 곳이 있는 상품 상세 (날짜별) ({start_date} ~ {end_date}{scope}) — 제품 {product_count}개 / 케이스 {case_count}행")
            print_result_any(out_df, output_dir=args.output_dir, prefix=f"Q5_RANGE_DETAIL_{mall}_{start_date}_to_{end_date}")
            print_friendly_suffix()
            last_result_df = out_df.copy()
            continue

        if intent == "Q5":
            if not mall:
                print("쇼핑몰명을 인식 못했어요. (엑셀에 존재하는 mall_name을 질문에 포함)\n")
                continue
            include_target_price = wants_target_price(q)
            include_diff = wants_diff(q)
            require_target_price = include_target_price or include_diff

            if start_date != end_date:
                print(f"[안내] 기간형 최저가 비교는 최신 날짜 기준으로 계산했습니다. (기준일: {end_date})")
                effective_batch_date = end_date
            else:
                effective_batch_date = batch_date

            out = q5_mall_not_cheapest(
                df_pc_all,
                effective_batch_date,
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
            table: pd.DataFrame = out["table"]

            print_friendly_prefix()
            print(
                f"[정형결과] {effective_batch_date} {scope}모든 과자 중 '{out['target_mall']}'이(가) 최저가가 아닌 제품: "
                f"제품 {out['product_count']}개 / 케이스 {out['case_count']}건 / 전체 {out['total_products']}개"
            )
            if table.empty:
                print("(해당 없음)")
            else:
                print(df_to_string_kr(table, index=False))

            ensure_dir(args.output_dir)
            mtag = manu if manu else "ALL"
            path = os.path.join(args.output_dir, f"Q5_not_cheapest_{out['target_mall']}_{mtag}_{out['batch_date']}.csv")
            table.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"\n[저장] CSV: {path}")
            print_friendly_suffix()
            last_result_df = table.copy()
            continue

        if intent == "Q6":
            if not mall:
                print("쇼핑몰명을 인식 못했어요. (엑셀에 존재하는 mall_name을 질문에 포함)\n")
                continue
            out = q6_mall_is_cheapest(df_pc_all, batch_date, mall, manufacturer=manu)
            if not out:
                print("데이터를 못 찾았어요.\n")
                continue
            scope = f"(제조사={manu}) " if manu else ""
            table = out["table"]
            print_friendly_prefix()
            print(
                f"[정형결과] {batch_date} {scope}'{mall}'이(가) 최저가인 제품: "
                f"제품 {out['product_count']}개 / 케이스 {out['case_count']}건 / 전체 {out['total_products']}개"
            )
            if table.empty:
                print("(해당 없음)")
            else:
                print(df_to_string_kr(table, index=False))
            print_friendly_suffix()
            last_result_df = table.copy()
            continue

        if intent == "Q7":
            if not mall:
                print("쇼핑몰명을 인식 못했어요. (엑셀에 존재하는 mall_name을 질문에 포함)\n")
                continue
            n = parse_top_n(q, default_n=10)
            metric = parse_metric_kor(q)
            ascending = parse_sort_direction(q, metric)

            out = q7_mall_metric_topn(df_pc_all, batch_date, mall, metric, n, ascending, manufacturer=manu)
            if not out:
                print("데이터를 못 찾았어요(지표/날짜/몰/제조사 확인).\n")
                continue

            direction = "낮은값 TOP" if ascending else "높은값 TOP"
            scope = f"(제조사={manu}) " if manu else ""
            print_friendly_prefix()
            print(f"[정형결과] {batch_date} {scope}{mall} {metric} {direction} {n}개")
            print(df_to_string_kr(out["table"], index=False))
            print_friendly_suffix()
            last_result_df = out["table"].copy()
            continue

        if intent == "Q8_TREND":
            if not pkey:
                print("제품을 인식 못했어요. (엑셀에 존재하는 제품코드/제품명이 질문에 포함되어야 함)\n")
                continue
            if requested_end_uncapped > today:
                print(f"[안내] 종료일 {requested_end_uncapped}은 데이터 범위를 벗어나 {today}까지로 계산했습니다.")
            out_df = q8_product_trend(df_pc_all, product_key=pkey, start_date=start_date, end_date=end_date, manufacturer=manu, mall=mall)
            if out_df is None:
                print("기간 내 데이터가 없습니다.\n")
                continue
            mall_scope = f", 몰={mall}" if mall else ""
            manu_scope = f", 제조사={manu}" if manu else ""
            title_range = start_date if start_date == end_date else f"{start_date} ~ {end_date}"
            print_friendly_prefix()
            print(f"[정형결과] {pkey} 가격 추이 ({title_range}{manu_scope}{mall_scope})")
            print_result_any(out_df, output_dir=args.output_dir, prefix=f"Q8_trend_{pkey}_{start_date}_to_{end_date}")
            print_friendly_suffix()
            last_result_df = out_df.copy()
            continue

        if intent == "Q7B_MANU_RANGE_TOPN":
            if not manu:
                print("제조사를 인식 못했어요. (엑셀에 존재하는 manufacturer 또는 자사/우리/당사 포함)\n")
                continue
            if requested_end_uncapped > today:
                print(f"[안내] 종료일 {requested_end_uncapped}은 데이터 범위를 벗어나 {today}까지로 계산했습니다.")
            n = parse_top_n(q, default_n=20)
            metric = parse_metric_kor(q)
            ascending = parse_sort_direction(q, metric)
            out_df = q7b_manufacturer_range_topn(df_pc_all, start_date, end_date, manu, metric, n, ascending)
            if out_df is None or len(out_df) == 0:
                print("기간 내 데이터가 없습니다.\n")
                continue
            direction = "낮은값 TOP" if ascending else "높은값 TOP"
            title_range = start_date if start_date == end_date else f"{start_date} ~ {end_date}"
            print_friendly_prefix()
            print(f"[정형결과] {manu} {metric} {direction} {n}개 ({title_range})")
            print(df_to_string_kr(out_df, index=False))
            print_friendly_suffix()
            last_result_df = out_df.copy()
            continue

        if intent == "Q7C_MANU_RANGE_TOPN_WITH_DATE":
            if not manu:
                print("제조사를 인식 못했어요. (엑셀에 존재하는 manufacturer 또는 자사/우리/당사 포함)\n")
                continue
            if requested_end_uncapped > today:
                print(f"[안내] 종료일 {requested_end_uncapped}은 데이터 범위를 벗어나 {today}까지로 계산했습니다.")
            n = parse_top_n(q, default_n=20)
            metric = parse_metric_kor(q)
            ascending = parse_sort_direction(q, metric)
            out_df = q7c_manufacturer_range_topn_with_date(df_pc_all, start_date, end_date, manu, metric, n, ascending)
            if out_df is None or len(out_df) == 0:
                print("기간 내 데이터가 없습니다.\n")
                continue
            direction = "낮은값 TOP" if ascending else "높은값 TOP"
            title_range = start_date if start_date == end_date else f"{start_date} ~ {end_date}"
            print_friendly_prefix()
            print(f"[정형결과] {manu} {metric} {direction} {n}개 - 날짜 포함 ({title_range})")
            print(df_to_string_kr(out_df, index=False))
            print_friendly_suffix()
            last_result_df = out_df.copy()
            continue

        if intent == "Q7D_MANU_RANGE_TOPN_BY_MALL":
            if not manu:
                print("제조사를 인식 못했어요. (엑셀에 존재하는 manufacturer 또는 자사/우리/당사 포함)\n")
                continue
            if requested_end_uncapped > today:
                print(f"[안내] 종료일 {requested_end_uncapped}은 데이터 범위를 벗어나 {today}까지로 계산했습니다.")
            n = parse_top_n(q, default_n=5)
            metric = parse_metric_kor(q)
            ascending = parse_sort_direction(q, metric)
            out_map = q7d_manufacturer_range_topn_by_mall(df_pc_all, start_date, end_date, manu, metric, n, ascending)
            if not out_map:
                print("기간 내 데이터가 없습니다.\n")
                continue
            direction = "낮은값 TOP" if ascending else "높은값 TOP"
            title_range = start_date if start_date == end_date else f"{start_date} ~ {end_date}"
            print_friendly_prefix()
            print(f"[정형결과] {manu} {metric} {direction} - 몰별 TOP {n} ({title_range})")
            for mall_name in sorted(out_map.keys()):
                print(f"\n--- {mall_name} ---")
                print(df_to_string_kr(out_map[mall_name], index=False))
            print_friendly_suffix()
            last_result_df = pd.concat(list(out_map.values()), ignore_index=True) if out_map else None
            continue

        if intent == "Q9_MALL_SUMMARY_TABLE":
            if not mall:
                print("쇼핑몰명을 인식 못했어요. (엑셀에 존재하는 mall_name을 질문에 포함)\n")
                continue
            if requested_end_uncapped > today:
                print(f"[안내] 종료일 {requested_end_uncapped}은 데이터 범위를 벗어나 {today}까지로 계산했습니다.")
            out_df = q9_mall_summary_table(df_pc_all, start_date, end_date, mall, manufacturer=manu)
            if out_df is None or len(out_df) == 0:
                print("기간 내 데이터가 없습니다.\n")
                continue
            title_range = start_date if start_date == end_date else f"{start_date} ~ {end_date}"
            scope = f", 제조사={manu}" if manu else ""
            print_friendly_prefix()
            print(f"[정형결과] {mall} 요약 가격표 ({title_range}{scope})")
            print_result_any(out_df, output_dir=args.output_dir, prefix=f"Q9_summary_{mall}_{start_date}_to_{end_date}")
            print_friendly_suffix()
            last_result_df = out_df.copy()
            continue

        if pkey is None and looks_like_product_query(q) and any(k in q for k in PRODUCT_QUERY_HINTS):
            print("제품을 인식 못했어요. 현재 데이터에 없는 제품명/제품코드일 수 있습니다.\n")
            continue

        if should_block_plan_fallback(q, slots):
            print("정형 규칙에 없는 위반/비교 질의입니다. 정형 라우터 보강이 필요합니다.\n")
            continue

        if not llm_plan_enabled:
            print("규칙에 없는 질문입니다. (LLM SAFE 폴백 비활성화)\n")
            continue

        print("[LLM SAFE 폴백] rule-based mini-plan 우선 실행 → 필요 시에만 LLM plan 생성")
        try:
            entities = {
                "pkey": pkey,
                "mall": mall,
                "mall_list": mall_list,
                "manu": manu,
                "manu_list": manu_list,
                "batch_date": batch_date,
            }

            raw_plan = build_rule_based_miniplan(
                q,
                batch_date=batch_date,
                start_date=start_date,
                end_date=end_date,
                mall=mall,
                manu=manu,
                pkey=pkey,
                mall_list=mall_list,
                manu_list=manu_list,
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
                mall_list=mall_list,
                manu_list=manu_list,
            )
            repaired_plan = enrich_plan_semantics(
                repaired_plan,
                question=q,
                start_date=start_date,
                end_date=end_date,
                mall=mall,
                manu=manu,
                pkey=pkey,
                mall_list=mall_list,
                manu_list=manu_list,
            )
            repaired_plan = validate_plan_against_question(
                q,
                repaired_plan,
                mall=mall,
                manu=manu,
                pkey=pkey,
                mall_list=mall_list,
                manu_list=manu_list,
                start_date=start_date,
                end_date=end_date,
            )

            maybe_need_llm_plan = False
            if str(repaired_plan.get("operation") or "none") == "none":
                if not repaired_plan.get("groupby") and not repaired_plan.get("aggregations"):
                    if not mall and not manu and not pkey and slots.get("has_compare_words"):
                        maybe_need_llm_plan = True

            if maybe_need_llm_plan:
                llm_plan_ctx = build_llm_plan_context(
                    df_pc_all, df_data_all, today, q, entities, slots, start_date, end_date
                )
                try:
                    llm_raw_plan = llm_generate_plan(
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
                        print("\n[LLM RAW PLAN]")
                        print(json.dumps(llm_raw_plan, ensure_ascii=False, indent=2))

                    repaired_plan = repair_plan_structure(
                        llm_raw_plan,
                        question=q,
                        batch_date=batch_date,
                        start_date=start_date,
                        end_date=end_date,
                        mall=mall,
                        manu=manu,
                        pkey=pkey,
                        mall_list=mall_list,
                        manu_list=manu_list,
                    )
                    repaired_plan = enrich_plan_semantics(
                        repaired_plan,
                        question=q,
                        start_date=start_date,
                        end_date=end_date,
                        mall=mall,
                        manu=manu,
                        pkey=pkey,
                        mall_list=mall_list,
                        manu_list=manu_list,
                    )
                    repaired_plan = validate_plan_against_question(
                        q,
                        repaired_plan,
                        mall=mall,
                        manu=manu,
                        pkey=pkey,
                        mall_list=mall_list,
                        manu_list=manu_list,
                        start_date=start_date,
                        end_date=end_date,
                    )
                except Exception as e:
                    print(f"[WARN] LLM JSON 생성 실패 → rule-based mini-plan 유지: {repr(e)}")

            if args.print_plan:
                print("\n[REPAIRED RAW PLAN]")
                print(json.dumps(repaired_plan, ensure_ascii=False, indent=2))

            plan = validate_and_normalize_plan(repaired_plan, df_pc_all=df_pc_all, df_data_all=df_data_all)

            if args.print_plan:
                print_normalized_plan(plan)

            out_df = execute_plan(plan, df_pc_all=df_pc_all, df_data_all=df_data_all)

            print_friendly_prefix()
            print("[LLM SAFE 결과]")
            print_result_any(out_df, output_dir=args.output_dir, prefix="LLM_SAFE")
            print_friendly_suffix()
            last_result_df = out_df.copy() if isinstance(out_df, pd.DataFrame) else None
        except Exception as e:
            print("LLM SAFE 폴백 실패:", repr(e))
            print("힌트: 질문에 날짜/몰/지표/제조사(또는 자사/우리/당사)를 포함하면 성공률이 더 올라갑니다.\n")

        gc.collect()


if __name__ == "__main__":
    main()