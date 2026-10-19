#!/usr/bin/env python3
"""
diagnose_clean.py — диагностика УЖЕ ОЧИЩЕННОГО датасета кофе.

Запуск:
    python diagnose_clean.py --input coffee_beans_clean.csv

Проверяет:
  1. Известные проблемы, которые clean_dataset.py должен был устранить
     (должно быть 0 — иначе регрессия).
  2. Консистентность новых полей
     (units_in_pack, weight_g_actual, segment, weight_source, is_outlier_price).
  3. Базовое качество данных (пропуски, дубликаты, типы).

Выход:
  diagnostic_clean_summary.csv — счётчики
  diagnostic_clean_issues.csv  — строки с флагами
  консольный отчёт (OK / WARN / FAIL по каждому пункту)
"""
import argparse
import json
import re

import pandas as pd

# ---------- ожидаемые значения новых полей ----------
VALID_SEGMENTS = {"consumer", "sample", "bulk"}
VALID_SOURCES = {"specs", "name_multipack", "name_single", "none"}
PRICE_KG_MIN, PRICE_KG_MAX = 200, 80000
PRICE_KG_GRAY_LOW = (200, 300)      # (min, max) — серая зона
PRICE_KG_GRAY_HIGH = (60000, 80000)

GROUND_RE = re.compile(r"\bмолот(ый|ого|ому|ым|ом|ая|ой|ую|ое|ые|ых)\b", re.I)
INSTANT_RE = re.compile(r"растворим", re.I)
CAPSULE_RE = re.compile(r"капсул", re.I)
DRIP_RE = re.compile(r"\bдрип", re.I)
THREE_IN_ONE_RE = re.compile(r"\d\s*в\s*1\b", re.I)

MULTIPACK_RE = re.compile(
    r"(\d+)\s*(?:шт\w*|пачк\w*|упаков\w*|уп)?\s*"
    r"(?:по|[xх×*])\s*"
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(?:грамм[а-я]*|килограмм[а-я]*|кг|kg|гр|gr|г|g)",
    re.I,
)
MULTIPACK_RE_REV = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:грамм[а-я]*|килограмм[а-я]*|кг|kg|гр|gr|г|g)\s*"
    r"[xх×*]\s*(\d+)",
    re.I,
)
SAME_N_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:грамм[а-я]*|килограмм[а-я]*|кг|kg|гр|gr|г|g)"
    r"\s*[\(\[]?\s*\1\s*(?:шт\w*|уп\w*|пачк\w*|комплект\w*)\b",
    re.I,
)
COUNT_RE = re.compile(r"(\d+)\s*(?:шт\w*|пачк\w*|упаков\w*|уп)\b", re.I)


def to_float(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (ValueError, TypeError):
        pass
    if isinstance(v, str):
        s = v.replace(",", ".").strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def is_blank(v):
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (ValueError, TypeError):
        pass
    return isinstance(v, str) and not v.strip()


# ---------- набор проверок ----------
def diagnose_row(row):
    """
    Возвращает dict {flag_name: bool, ...}.
    Все флаги — то, что ДОЛЖНО БЫТЬ ЛОЖЬЮ после очистки.
    """
    f = {}

    name = row.get("name")
    brand = row.get("brand")
    price = to_float(row.get("price"))
    ppk = to_float(row.get("price_per_kg"))
    wg = to_float(row.get("weight_g"))
    wa = to_float(row.get("weight_g_actual"))
    units = to_float(row.get("units_in_pack"))
    segment = row.get("segment")
    source = row.get("weight_source")
    outlier = row.get("is_outlier_price")

    # --- вес ---
    f["weight_actual_null"] = wa is None
    f["weight_actual_le0"] = wa is not None and wa <= 0
    f["weight_actual_gt_10kg_not_bulk"] = (
        wa is not None and wa > 10000 and segment != "bulk"
    )
    f["weight_actual_lt_100g_not_sample"] = (
        wa is not None and 0 < wa < 100 and segment != "sample"
    )
    # weight_g "некруглый" (должен был быть rounded/заменён)
    f["weight_g_off_round"] = (
        wg is not None and wg > 0
        and wg % 10 != 0
        and int(wg) not in {227, 907, 2270, 340, 375, 150, 250, 350, 450, 750}
        and wa is not None
        and wg == wa  # если clean его не поправил, а оставил
    )

    # --- units ---
    f["units_eq_0"] = units is not None and units == 0
    f["units_gt_10"] = units is not None and units > 10
    # units>1 без мультипака в названии — регрессия
        # units>1 без мультипака в названии — регрессия
    if units is not None and units > 1 and isinstance(name, str):
        mp_found = (
            MULTIPACK_RE.search(name)
            or MULTIPACK_RE_REV.search(name)
            or SAME_N_RE.search(name)
        )
        if not mp_found:
            # а может это просто "N штук" как счётчик?
            counts = [int(x) for x in COUNT_RE.findall(name)]
            if units in counts:
                mp_found = True
        f["units_gt1_no_mp_in_name"] = not mp_found
    else:
        f["units_gt1_no_mp_in_name"] = False

    # --- цена ---
    f["price_null"] = price is None
    f["price_le0"] = price is not None and price <= 0
    f["price_per_kg_null"] = ppk is None
    if ppk is not None:
        f["price_per_kg_outside_200_80000"] = not (PRICE_KG_MIN < ppk < PRICE_KG_MAX)
        f["price_per_kg_in_gray_zone"] = (
            PRICE_KG_GRAY_LOW[0] < ppk < PRICE_KG_GRAY_LOW[1]
            or PRICE_KG_GRAY_HIGH[0] < ppk < PRICE_KG_GRAY_HIGH[1]
        )
        # консистентность: если помечено outlier — должно быть вне диапазона
        f["outlier_flag_mismatch"] = (
            (outlier is True and PRICE_KG_MIN < ppk < PRICE_KG_MAX and
             not (PRICE_KG_GRAY_LOW[0] < ppk < PRICE_KG_GRAY_LOW[1] or
                  PRICE_KG_GRAY_HIGH[0] < ppk < PRICE_KG_GRAY_HIGH[1]))
            or
            (outlier is False and not (PRICE_KG_MIN < ppk < PRICE_KG_MAX))
        )
        # пересчёт: price/weight_g_actual*1000 ≈ price_per_kg
        if price is not None and wa not in (None, 0):
            recomputed = round(price / wa * 1000)
            f["price_per_kg_mismatch"] = abs(recomputed - ppk) > 1
        else:
            f["price_per_kg_mismatch"] = False
    else:
        f["price_per_kg_outside_200_80000"] = False
        f["price_per_kg_in_gray_zone"] = False
        f["outlier_flag_mismatch"] = False
        f["price_per_kg_mismatch"] = False

    # --- текст: не-зерно (должно быть 0) ---
    if isinstance(name, str):
        f["name_ground"] = bool(GROUND_RE.search(name))
        f["name_instant"] = bool(INSTANT_RE.search(name))
        f["name_capsule"] = bool(CAPSULE_RE.search(name))
        f["name_drip"] = bool(DRIP_RE.search(name))
        f["name_3in1"] = bool(THREE_IN_ONE_RE.search(name))
    else:
        f["name_ground"] = f["name_instant"] = False
        f["name_capsule"] = f["name_drip"] = f["name_3in1"] = False

    # --- бренд ---
    f["brand_null"] = is_blank(brand)
    if isinstance(brand, str) and brand.strip():
        # бренд "как есть" в нижнем регистре и/или весь кириллицей в нижнем —
        # подозрение, что нормализация не применилась
        f["brand_lowercase"] = brand == brand.lower() and brand != brand.upper()
        f["brand_whitespace"] = brand != brand.strip()
    else:
        f["brand_lowercase"] = False
        f["brand_whitespace"] = False

    # --- метаданные ---
    f["product_type_not_beans"] = row.get("product_type") != "Кофе в зернах"
    f["parse_status_bad"] = row.get("parse_status") not in ("ok", "ok_partial")
    f["availability_not_instock"] = row.get("availability") != "InStock"
    f["currency_not_rub"] = str(row.get("currency") or "").upper() != "RUB"
    f["specs_json_null"] = is_blank(row.get("specs_json"))

    # --- новые поля: валидность ---
    f["segment_invalid"] = segment not in VALID_SEGMENTS
    f["weight_source_invalid"] = source not in VALID_SOURCES

    # --- заполненность ключевых полей ---
    f["rating_null"] = is_blank(row.get("rating"))
    f["reviews_count_null"] = is_blank(row.get("reviews_count"))
    f["seller_name_null"] = is_blank(row.get("seller_name"))

    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/coffee_beans_clean.csv")
    ap.add_argument("--out-summary", default="diagnostic_clean_summary.csv")
    ap.add_argument("--out-issues", default="diagnostic_clean_issues.csv")
    args = ap.parse_args()

    print(f"Читаю {args.input} ...")
    df = pd.read_csv(args.input, dtype=str)
    print(f"  строк: {len(df)}, колонок: {len(df.columns)}")

    print("Прогоняю проверки ...")
    records = []
    for _, row in df.iterrows():
        flags = diagnose_row(row)
        rec = {
            "sku": row.get("sku"),
            "name": row.get("name"),
            "brand": row.get("brand"),
            "price": row.get("price"),
            "weight_g": row.get("weight_g"),
            "weight_g_actual": row.get("weight_g_actual"),
            "units_in_pack": row.get("units_in_pack"),
            "price_per_kg": row.get("price_per_kg"),
            "weight_source": row.get("weight_source"),
            "segment": row.get("segment"),
            "is_outlier_price": row.get("is_outlier_price"),
        }
        rec["n_issues"] = sum(flags.values())
        rec["issues"] = ",".join(k for k, v in flags.items() if v)
        rec.update(flags)
        records.append(rec)
    diag = pd.DataFrame(records)

    # ---------- summary ----------
    rows_summary = []
    for col in diag.columns:
        if col in ("sku", "name", "brand", "price", "weight_g",
                   "weight_g_actual", "units_in_pack", "price_per_kg",
                   "weight_source", "segment", "is_outlier_price",
                   "n_issues", "issues"):
            continue
        n = int(diag[col].sum())
        rows_summary.append({"check": col, "count": n,
                             "pct": round(100 * n / len(diag), 2)})
    summary = pd.DataFrame(rows_summary).sort_values("count", ascending=False)
    summary.to_csv(args.out_summary, index=False, encoding="utf-8-sig")

    # ---------- issues ----------
    issues = diag[diag["n_issues"] > 0].copy()
    priority = ["sku", "name", "brand", "price", "weight_g",
                "weight_g_actual", "units_in_pack", "price_per_kg",
                "weight_source", "segment", "is_outlier_price",
                "n_issues", "issues"]
    other = [c for c in issues.columns if c not in priority]
    issues = issues[priority + other].sort_values("n_issues", ascending=False)
    issues.to_csv(args.out_issues, index=False, encoding="utf-8-sig")

    # ---------- консоль ----------
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ПРОВЕРОК")
    print("=" * 70)

    # строгие проверки — должны быть 0 после очистки
    strict_zero = [
        "weight_actual_null", "weight_actual_le0",
        "weight_actual_gt_10kg_not_bulk", "weight_actual_lt_100g_not_sample",
        "units_eq_0", "units_gt_10", "units_gt1_no_mp_in_name",
        "price_null", "price_le0", "price_per_kg_null",
        "price_per_kg_mismatch", "outlier_flag_mismatch",
        "name_ground", "name_instant", "name_capsule", "name_drip", "name_3in1",
        "product_type_not_beans", "parse_status_bad",
        "availability_not_instock", "currency_not_rub",
        "segment_invalid", "weight_source_invalid",
    ]
    print("\n[ДОЛЖНО БЫТЬ 0 — регрессии]")
    fail = 0
    for k in strict_zero:
        n = int(diag[k].sum())
        status = "OK  " if n == 0 else "FAIL"
        if n > 0:
            fail += 1
        print(f"  {status} {k:40s} {n:5d}")

    # ожидаемые (не 0, но норма)
    print("\n[Ожидаемые ненулевые значения — не регрессия]")
    for k in ("price_per_kg_outside_200_80000", "price_per_kg_in_gray_zone",
              "brand_null", "brand_lowercase",
              "rating_null", "reviews_count_null", "seller_name_null",
              "specs_json_null", "weight_g_off_round"):
        n = int(diag[k].sum())
        print(f"  --- {k:40s} {n:5d}")

    # ---------- дубликаты ----------
    print("\n[Дубликаты]")
    if "sku" in diag.columns:
        dup_sku = diag[diag["sku"].duplicated(keep=False)].sort_values("sku")
        print(f"  по sku:            {diag['sku'].duplicated().sum()}")
    if all(c in diag.columns for c in ("name", "brand", "weight_g_actual")):
        dup_full = df[df.duplicated(subset=["name", "brand", "weight_g_actual"],
                                    keep=False)]
        print(f"  name+brand+weight: {len(dup_full)} строк "
              f"(уникальных групп: {dup_full.groupby(['name','brand','weight_g_actual']).ngroups})")

    # ---------- распределения ----------
    print("\n[segment]")
    print(df["segment"].value_counts(dropna=False).to_string())
    print("\n[weight_source]")
    print(df["weight_source"].value_counts(dropna=False).to_string())
    print("\n[is_outlier_price]")
    print(df["is_outlier_price"].value_counts(dropna=False).to_string())

    print("\n" + "=" * 70)
    if fail == 0:
        print("ИТОГ: все строгие проверки = 0. Датасет чист.")
    else:
        print(f"ИТОГ: {fail} строгих проверок с ненулевым счётчиком — смотри "
              f"{args.out_issues}")
    print("=" * 70)

    print(f"\nSummary -> {args.out_summary}")
    print(f"Issues  -> {args.out_issues} ({len(issues)} строк с флагами)")


if __name__ == "__main__":
    main()