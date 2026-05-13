#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import json
import argparse

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
    MY_MANUFACTURER,
    PRODUCT_QUERY_HINTS,
    build_catalog,
    discover_xlsx_files,
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
    print_debug_json,
    print_result_any,
    resolve_my_manufacturer,
    rf_process,
    today_from_df,
    upsert_price_compare,
    upsert_reviews_digest,
)

from poc_snacks_logic import (
    build_llm_plan_context,
    build_rule_based_miniplan,
    enrich_plan_semantics,
    execute_plan,
    extract_semantic_slots,
    infer_time_mode,
    init_local_llm,
    llm_generate_plan,
    print_normalized_plan,
    repair_plan_structure,
    should_block_plan_fallback,
    validate_and_normalize_plan,
    validate_plan_against_question,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Snacks POC LLM-only Router (special operations)")

    p.add_argument("--folder", default="/home/siwasoft/gsllm/xlsx_input", help="xlsx folder path")
    p.add_argument("--file-regex", default=r"^\d{4}-\d{2}-\d{2}_snacks.*\.xlsx$", help="xlsx basename regex")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="output directory for CSV")

    p.add_argument("--enable-chroma", action="store_true", help="enable chroma upsert")
    p.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH)
    p.add_argument("--coll-summary", default=DEFAULT_COLL_SUMMARY)
    p.add_argument("--coll-reviews", default=DEFAULT_COLL_REVIEWS)
    p.add_argument("--disable-reviews-digest", action="store_true", help="disable reviews digest upsert")

    p.add_argument("--enable-llm", action="store_true", help="enable LLM fallback")
    p.add_argument("--llm-model-path", default=DEFAULT_LLM_MODEL_PATH)
    p.add_argument("--llm-n-ctx", type=int, default=DEFAULT_LLM_N_CTX)
    p.add_argument("--llm-max-tokens", type=int, default=DEFAULT_LLM_MAX_TOKENS)
    p.add_argument("--llm-temperature", type=float, default=DEFAULT_LLM_TEMPERATURE)
    p.add_argument("--llm-threads", type=int, default=DEFAULT_LLM_THREADS)
    p.add_argument("--llm-gpu-layers", type=int, default=DEFAULT_LLM_GPU_LAYERS)

    p.add_argument("--print-plan", action="store_true", help="print RAW/REPAIRED/NORMALIZED plan")
    p.add_argument("--disable-rule-miniplan", action="store_true", help="disable rule-based fallback mini-plan")
    p.add_argument("--trace-route", action="store_true", help="print lightweight route debug")
    p.add_argument("--allow-blocked-plan", action="store_true", help="even if business compare/violation pattern is detected, still try LLM plan")

    p.set_defaults(
        enable_llm=DEFAULT_ENABLE_LLM,
        print_plan=DEFAULT_PRINT_PLAN,
    )
    return p


def main():
    args = build_argparser().parse_args()

    print("=== Snacks POC LLM-only Router (special operations) ===")
    print(f"(자사 제조사 고정) MY_MANUFACTURER = {MY_MANUFACTURER}")
    print(f"[DEFAULT LLM MODEL] {args.llm_model_path}")
    print(
        f"[DEFAULT LLM CTX] n_ctx={args.llm_n_ctx}, "
        f"gpu_layers={args.llm_gpu_layers}, "
        f"print_plan={args.print_plan}, enable_llm={args.enable_llm}"
    )

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
            upsert_price_compare(col_sum, df_pc_all, version="llm_only_special")
            if not args.disable_reviews_digest:
                upsert_reviews_digest(col_rev, df_data_all, version="llm_only_special")
            print("Chroma 업서트 완료")
        except Exception as e:
            print("Chroma 오류(무시하고 계속):", repr(e))

    today = today_from_df(df_pc_all)
    default_start = min_date_from_df(df_pc_all, fallback=today)
    print(f"\n현재 합본 기준 today(batch_date) = {today} (default_start={default_start})\n")

    if args.enable_llm:
        print("[LLM] 모델 로드 확인 중...")
        try:
            init_local_llm(
                model_path=args.llm_model_path,
                n_ctx=args.llm_n_ctx,
                n_threads=args.llm_threads,
                n_gpu_layers=args.llm_gpu_layers,
            )
            print("[LLM] 모델 로드 성공\n")
        except Exception as e:
            print("[LLM] 모델 로드 실패:", repr(e))
            return

    if not args.enable_llm:
        print("[주의] enable_llm=False 상태입니다. 이 파일은 사실상 LLM 전용 테스트용입니다.\n")
    else:
        print("[LLM] 모든 질문을 LLM PandasPlan으로 처리합니다.")
        if args.disable_rule_miniplan:
            print("[Fallback] 규칙 기반 mini-plan 비활성화\n")
        else:
            print("[Fallback] LLM JSON 실패 시 규칙 기반 mini-plan 1회 대체\n")

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
        plan_blocked = should_block_plan_fallback(q, slots)

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
            "mode": "LLM_ONLY",
            "q": q,
            "entities": entities_for_debug,
            "slots": slots,
            "view_source": slots.get("view_source", "unknown"),
            "time_mode": time_mode,
            "plan_blocked_by_business_rule": plan_blocked,
        })

        if args.trace_route:
            print(
                f"[TRACE] pkey={pkey} mall={mall} manu={manu} "
                f"start_date={start_date} end_date={end_date} blocked={plan_blocked}"
            )

        if plan_blocked and not args.allow_blocked_plan:
            print("[주의] 위반/비교형 질문으로 감지되었습니다. 지금은 special operation으로 계속 시도합니다.\n")

        if pkey is None and looks_like_product_query(q) and any(k in q for k in PRODUCT_QUERY_HINTS):
            print("[참고] 제품처럼 보이지만 현재 카탈로그에서 제품을 인식하지 못했습니다. 그래도 LLM plan은 계속 시도합니다.\n")

        entities = {
            "pkey": pkey,
            "mall": mall,
            "mall_list": mall_list,
            "manu": manu,
            "manu_list": manu_list,
            "batch_date": batch_date,
        }

        print("[LLM-ONLY] PandasPlan JSON plan 생성 → 실행 중...")

        try:
            llm_plan_ctx = build_llm_plan_context(
                df_pc_all=df_pc_all,
                df_data_all=df_data_all,
                today=today,
                question=q,
                entities=entities,
                slots=slots,
                start_date=start_date,
                end_date=end_date,
            )

            used_rule_miniplan = False

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
                if args.disable_rule_miniplan:
                    raise RuntimeError(f"LLM JSON 생성 실패(rule miniplan disabled): {repr(e)}")
                print(f"[WARN] LLM JSON 생성 실패 → 규칙 기반 mini-plan으로 대체: {repr(e)}")
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
                used_rule_miniplan = True
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

            if args.print_plan:
                print("\n[REPAIRED RAW PLAN]")
                print(json.dumps(repaired_plan, ensure_ascii=False, indent=2))

            plan = validate_and_normalize_plan(
                repaired_plan,
                df_pc_all=df_pc_all,
                df_data_all=df_data_all,
            )

            if args.print_plan:
                print_normalized_plan(plan)

            out_df = execute_plan(
                plan,
                df_pc_all=df_pc_all,
                df_data_all=df_data_all,
            )

            print("\n[LLM-ONLY 결과]")
            if used_rule_miniplan:
                print("(참고: 이번 결과는 LLM JSON 실패 후 rule mini-plan 대체 실행 결과입니다.)")
            print_result_any(out_df, output_dir=args.output_dir, prefix="LLM_ONLY")
            print()

        except Exception as e:
            print("LLM-ONLY 실행 실패:", repr(e))
            print("힌트: 날짜/몰/지표/제조사(또는 자사/우리/당사)를 더 명시하면 성공률이 올라갑니다.\n")

        gc.collect()


if __name__ == "__main__":
    main()