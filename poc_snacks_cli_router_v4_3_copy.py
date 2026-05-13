#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc

from poc_snacks_shared import (
    DEFAULT_ENABLE_LLM,
    DEFAULT_LLM_GPU_LAYERS,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL_PATH,
    DEFAULT_LLM_N_CTX,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_THREADS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PRINT_PLAN,
    MY_MANUFACTURER,
    build_catalog,
    discover_xlsx_files,
    extract_target_mall,
    load_excels_multi,
    min_date_from_df,
    normalize_batch_date_series,
    normalize_range_separators,
    parse_batch_date,
    parse_date_range,
    parse_dates_list,
    parse_product_key,
    parse_requested_date_range_uncapped,
    print_debug_json,
    print_result_any,
    today_from_df,
)
from poc_snacks_logic import (
    compare_with_mall,
    detect_intent,
    extract_semantic_slots,
    infer_time_mode,
    llm_generate_compare_spec,
    parse_compare_direction,
    product_compare,
    q1_product_all_offers,
    q8_product_trend,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Snacks POC CLI Router (웹크롤링_YYYYMMDD 전용)")
    p.add_argument("--folder", default="/home/siwasoft/gsllm/xlsx_input", help="xlsx folder path")
    p.add_argument("--file-regex", default=r"(snack_crawling|웹크롤링)_\d{8}\.xlsx$", help="xlsx basename regex")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--enable-llm", action="store_true", help="enable LLM fallback")
    p.add_argument("--llm-model-path", default=DEFAULT_LLM_MODEL_PATH)
    p.add_argument("--llm-n-ctx", type=int, default=DEFAULT_LLM_N_CTX)
    p.add_argument("--llm-max-tokens", type=int, default=DEFAULT_LLM_MAX_TOKENS)
    p.add_argument("--llm-temperature", type=float, default=DEFAULT_LLM_TEMPERATURE)
    p.add_argument("--llm-threads", type=int, default=DEFAULT_LLM_THREADS)
    p.add_argument("--llm-gpu-layers", type=int, default=DEFAULT_LLM_GPU_LAYERS)
    p.add_argument("--print-plan", action="store_true")
    p.set_defaults(enable_llm=DEFAULT_ENABLE_LLM, print_plan=DEFAULT_PRINT_PLAN)
    return p


def main():
    args = build_argparser().parse_args()

    print("=== Snacks POC CLI Router (웹크롤링_YYYYMMDD 전용) ===")
    print(f"(자사 제조사 고정) MY_MANUFACTURER = {MY_MANUFACTURER}")
    print(f"[DEFAULT LLM MODEL] {args.llm_model_path}")
    print(f"[DEFAULT LLM CTX] n_ctx={args.llm_n_ctx}, gpu_layers={args.llm_gpu_layers}, print_plan={args.print_plan}, enable_llm={args.enable_llm}")

    files = discover_xlsx_files(args.folder, args.file_regex)
    if not files:
        print("xlsx 파일을 찾지 못했습니다.")
        print(f"- folder: {args.folder}")
        print(f"- file_regex: {args.file_regex}")
        return

    df_data_all, df_compare_all = load_excels_multi(args.folder, files)
    if len(df_data_all) == 0:
        print("로드된 데이터가 비어 있습니다.")
        return

    df_data_all["batch_date"] = normalize_batch_date_series(df_data_all["batch_date"])
    if len(df_compare_all):
        df_compare_all["batch_date"] = normalize_batch_date_series(df_compare_all["batch_date"])

    print(f"로드 완료({len(files)}개 합본): DATA={len(df_data_all):,}행, COMPARE={len(df_compare_all):,}행")
    if len(df_compare_all):
        print(f"날짜 범위: {df_compare_all['batch_date'].min()} ~ {df_compare_all['batch_date'].max()}")
    else:
        print("날짜 범위: COMPARE 매칭 결과 없음")

    catalog = build_catalog(df_compare_all, df_data_all)
    print(f"[카탈로그] 몰={len(catalog['malls'])}개, 제조사={len(catalog['manufacturers'])}개, 제품={len(catalog['product_key_set'])}개")

    today = today_from_df(df_compare_all if len(df_compare_all) else df_data_all)
    default_start = min_date_from_df(df_compare_all if len(df_compare_all) else df_data_all, fallback=today)
    print(f"\n현재 합본 기준 today(batch_date) = {today} (default_start={default_start})\n")
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
        dates_list = parse_dates_list(q, default_date=today)
        batch_date = dates_list[0] if dates_list else parse_batch_date(q, default_date=today)
        start_date, end_date = parse_date_range(q, default_end=today, default_start=default_start)
        requested_start_uncapped, requested_end_uncapped = parse_requested_date_range_uncapped(q, default_end=today)
        slots = extract_semantic_slots(q)
        time_mode = infer_time_mode(q, start_date, end_date, slots)
        intent_before = detect_intent(q, has_product=bool(pkey), has_mall=bool(mall), start_date=start_date, end_date=end_date)
        intent_after = intent_before
        llm_spec = None

        if args.enable_llm and (intent_before == "UNKNOWN" or (intent_before == "COMPARE_WITH_MALL" and not mall)):
            llm_spec = llm_generate_compare_spec(
                q,
                today=today,
                start_date=start_date,
                end_date=end_date,
                model_path=args.llm_model_path,
                n_ctx=args.llm_n_ctx,
                max_tokens=min(220, args.llm_max_tokens),
                temperature=min(0.2, args.llm_temperature),
                n_threads=args.llm_threads,
                n_gpu_layers=args.llm_gpu_layers,
                mall_candidates=catalog["malls"],
                has_product=bool(pkey),
            )
            if llm_spec:
                intent_after = str(llm_spec.get("intent") or intent_before)
                if not mall and llm_spec.get("compare_mall"):
                    mall = str(llm_spec.get("compare_mall"))

        print_debug_json({
            "event": "parsed_question",
            "q": q,
            "entities": {
                "pkey": pkey,
                "mall": mall,
                "manu": None,
                "batch_date": batch_date,
                "start_date": start_date,
                "end_date": end_date,
                "requested_start_uncapped": requested_start_uncapped,
                "requested_end_uncapped": requested_end_uncapped,
            },
            "slots": slots,
            "time_mode": time_mode,
            "intent_before_llm": intent_before,
            "intent_after_llm": intent_after,
            "llm_spec": llm_spec,
        })

        intent = intent_after

        if intent == "COMPARE_WITH_MALL":
            if not mall or mall == "쿠팡해태":
                print("비교 몰을 인식 못했어요. 예: 쿠팡, 11번가, 과자를더하다\n")
                continue
            direction = parse_compare_direction(q, mall)
            if llm_spec and llm_spec.get("direction") and llm_spec.get("direction") != "all":
                direction = str(llm_spec.get("direction"))
            summary_only = not slots.get("wants_detail", False)

            print("[DEBUG_COMPARE] mall=", mall, "direction=", direction, "rows_before=", len(df_compare_all))
            tmp_dbg = compare_with_mall(df_compare_all, start_date=start_date, end_date=end_date, compare_mall=mall, direction="all", product_key=pkey, summary_only=False)
            print("[DEBUG_COMPARE] rows_all_direction=", len(tmp_dbg))
            print("[DEBUG_COMPARE] diff<0=", int((tmp_dbg["diff_per_10g"] < 0).sum()) if len(tmp_dbg) else 0, "diff>0=", int((tmp_dbg["diff_per_10g"] > 0).sum()) if len(tmp_dbg) else 0)

            out_df = compare_with_mall(df_compare_all, start_date=start_date, end_date=end_date, compare_mall=mall, direction=direction, product_key=pkey, summary_only=summary_only)
            print(f"[비교결과] 쿠팡해태 vs {mall} ({start_date if start_date == end_date else f'{start_date} ~ {end_date}'})")
            print_result_any(out_df, output_dir=args.output_dir, prefix="COMPARE_WITH_MALL")
            continue

        if intent == "PRODUCT_COMPARE":
            if not pkey:
                print("제품을 인식 못했어요.\n")
                continue
            if len(df_compare_all) == 0:
                print("비교 데이터가 없습니다.\n")
                continue
            cur = product_compare(df_compare_all, product_key=pkey, start_date=batch_date, end_date=batch_date)
            if len(cur) == 0:
                print("데이터를 못 찾았어요.\n")
                continue
            print(f"[비교결과] {batch_date} 제품 비교")
            print_result_any(cur, output_dir=args.output_dir, prefix="PRODUCT_COMPARE")
            continue

        if intent == "Q8_TREND":
            if not pkey:
                print("제품을 인식 못했어요.\n")
                continue
            out_df = q8_product_trend(df_compare_all, product_key=pkey, start_date=start_date, end_date=end_date)
            if out_df is None or len(out_df) == 0:
                print("기간 내 데이터가 없습니다.\n")
                continue
            print(f"[추이결과] {start_date} ~ {end_date}")
            print_result_any(out_df, output_dir=args.output_dir, prefix="PRODUCT_TREND")
            continue

        if intent == "PRODUCT_RAW":
            if not pkey:
                print("제품을 인식 못했어요.\n")
                continue
            offers = q1_product_all_offers(df_data_all, product_key=pkey, batch_date=batch_date)
            if offers is None or len(offers) == 0:
                print("데이터를 못 찾았어요.\n")
                continue
            print(f"[원본결과] {batch_date} 제품 원본")
            print_result_any(offers, output_dir=args.output_dir, prefix="PRODUCT_RAW")
            continue

        print("질문을 해석하지 못했어요. 예: '오늘 쿠팡보다 비싼 제품 모두 알려줘', '오예스 오리지널 840g 1개 비교'\n")
        gc.collect()


if __name__ == "__main__":
    main()
