import csv
import random
import re
from datetime import datetime, timedelta, date

# ============================================================
# 여러 날짜 더미데이터 생성 (DATA_YYYY-MM-DD.csv / PRICE_COMPARE_YYYY-MM-DD.csv)
# - 기간: 2026-02-26 ~ 2026-03-10
# ============================================================

# ✅ 기간 자동 생성 (2/26 ~ 3/10)
START_DATE = "2026-02-26"
END_DATE = "2026-03-10"

def make_dates(start_ymd: str, end_ymd: str) -> list[str]:
    s = datetime.strptime(start_ymd, "%Y-%m-%d").date()
    e = datetime.strptime(end_ymd, "%Y-%m-%d").date()
    if s > e:
        s, e = e, s
    out = []
    cur = s
    while cur <= e:
        out.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out

dates = make_dates(START_DATE, END_DATE)

# ✅ 제조사명 변경
manufacturers = ["해태제과", "롯데", "오리온", "농심", "빙그레"]
products_per_manu = 50

# ✅ 제조사 -> 제품코드 알파벳 매핑
MANU_TO_CODE_PREFIX = {
    "해태제과": "H",
    "롯데": "L",
    "오리온": "O",
    "농심": "N",
    "빙그레": "B",
}

# ✅ 제품명 접두어(브랜드) 매핑
MANU_TO_NAME_PREFIX = {
    "해태제과": "해태",
    "롯데": "롯데",
    "오리온": "오리온",
    "농심": "농심",
    "빙그레": "빙그레",
}

# 제품 리스트 생성: (manufacturer, product_key, product_name)
products = []
for manu in manufacturers:
    code_prefix = MANU_TO_CODE_PREFIX[manu]
    name_prefix = MANU_TO_NAME_PREFIX[manu]
    for i in range(products_per_manu):
        pkey = f"{code_prefix}{str(i).zfill(3)}"
        pname = f"{name_prefix}과자{str(i).zfill(3)}"
        products.append((manu, pkey, pname))

# ✅ 마켓(쇼핑몰) 이름 변경
malls = ["SSG", "네이버쇼핑", "컬리", "쿠팡", "G마켓"]
offers_per_mall = 30

comment_pool = [
    "가성비 좋아요.", "배송 빨라요.", "포장 깔끔해요.", "맛있어요.", "재구매 의사 있어요.",
    "바삭하고 고소해요.", "유통기한 넉넉해요.", "아이들 간식으로 좋아요.", "가격 만족합니다.", "무난한 맛이에요.",
    "부스러기 적어요.", "기름지지 않아요.", "식감이 좋아요.", "커피랑 잘 어울려요.", "설명 그대로예요."
]

# 몰별 기준가(2/26 기준)
mall_base = {"SSG": 850, "네이버쇼핑": 830, "컬리": 870, "쿠팡": 825, "G마켓": 840}

# 2/26 기준 시작시간
base_time_0226 = datetime(2026, 2, 26, 9, 0, 0)

# ---------- 추이 제어 파라미터 ----------
DAY_DRIFT_BASE = 1.5
MALL_DRIFT_PER_DAY = {"SSG": 0.8, "네이버쇼핑": 0.4, "컬리": 1.0, "쿠팡": 0.6, "G마켓": 0.7}
MANU_DRIFT_PER_DAY = {"해태제과": 0.2, "롯데": 0.5, "오리온": 0.1, "농심": 0.3, "빙그레": 0.4}

RANK_STEP = 10
PRICE_NOISE_MAX = 2
PRODUCT_MALL_OFFSET_RANGE = 40
COMMENT_VARIATION = True


def date_to_day_index(ds: str) -> int:
    y, m, d = map(int, ds.split("-"))
    base = date(2026, 2, 26)
    cur = date(y, m, d)
    return (cur - base).days


def seed_for_date(ds: str) -> int:
    return int(ds.replace("-", "")) + 11


def stable_shuffle_sample(rng: random.Random, pool, k: int):
    arr = list(pool)
    rng.shuffle(arr)
    return arr[:k]


def stable_hash_seed(s: str) -> int:
    acc = 0
    for i, ch in enumerate(s, start=1):
        acc = (acc + (ord(ch) * i)) % 2147483647
    return acc


for batch_date in dates:
    day_idx = date_to_day_index(batch_date)
    rng = random.Random(seed_for_date(batch_date))
    base_time = base_time_0226 + timedelta(days=day_idx)

    summary_acc = {(pkey, mall): [] for _, pkey, _ in products for mall in malls}

    data_path = f"DATA_{batch_date}.csv"
    pc_path = f"PRICE_COMPARE_{batch_date}.csv"

    with open(data_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "batch_date", "mall_name", "manufacturer",
            "product_key", "product_name",
            "rank", "price", "date", "comments_top5"
        ])

        for p_idx, (manu, pkey, pname) in enumerate(products, start=1):
            product_offset = (p_idx % 50) * 7
            prod_num = int(re.sub(r"\D", "", pkey) or "0")
            product_drift_per_day = (prod_num % 7) * 0.1

            for m_idx, mall in enumerate(malls):
                pm_seed = stable_hash_seed(f"{pkey}|{mall}")
                pm_rng = random.Random(pm_seed)
                product_mall_offset = pm_rng.randint(-PRODUCT_MALL_OFFSET_RANGE, PRODUCT_MALL_OFFSET_RANGE)

                mall_time = base_time + timedelta(minutes=(p_idx - 1) * 2 + m_idx * 5)

                drift = (
                    day_idx * DAY_DRIFT_BASE
                    + day_idx * MALL_DRIFT_PER_DAY.get(mall, 0.0)
                    + day_idx * MANU_DRIFT_PER_DAY.get(manu, 0.0)
                    + day_idx * product_drift_per_day
                )

                base = mall_base[mall] + product_offset + product_mall_offset + drift

                for r in range(1, offers_per_mall + 1):
                    noise = rng.randint(0, PRICE_NOISE_MAX)
                    price = int(round(base + (r - 1) * RANK_STEP + noise))

                    if COMMENT_VARIATION:
                        c5 = " / ".join(stable_shuffle_sample(rng, comment_pool, 5))
                    else:
                        c5 = " / ".join(random.sample(comment_pool, 5))

                    dt = mall_time + timedelta(seconds=(r - 1) * 10)

                    w.writerow([
                        batch_date, mall, manu,
                        pkey, pname,
                        r, price,
                        dt.strftime("%Y-%m-%d %H:%M:%S"),
                        c5
                    ])
                    summary_acc[(pkey, mall)].append(price)

    with open(pc_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "batch_date", "mall_name", "manufacturer",
            "product_key", "product_name",
            "min_price", "avg_price", "max_price", "embedding_text"
        ])

        for (manu, pkey, pname) in products:
            for mall in malls:
                prices = summary_acc[(pkey, mall)]
                mn = min(prices)
                mx = max(prices)
                avg = round(sum(prices) / len(prices), 2)

                emb = (
                    f"[가격비교] 배치일={batch_date} | 쇼핑몰={mall} | 제조사={manu} | 제품={pname}({pkey})\n"
                    f"상위 {offers_per_mall}개(최저가순) 기준: 최저가={mn}원, 평균가={avg}원, 최고가={mx}원.\n"
                    f"참고: 비교/추이는 정형 데이터(min/avg/max 및 rank별 price)로 계산. "
                    f"텍스트(제조사/제품/요약)는 검색(임베딩)용."
                )

                w.writerow([batch_date, mall, manu, pkey, pname, mn, avg, mx, emb])

    print(f"완료: {data_path} (37,500행), {pc_path} (1,250행) 생성됨")

print("\n전체 완료.")