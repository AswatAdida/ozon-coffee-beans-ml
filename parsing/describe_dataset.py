#!/usr/bin/env python3
"""
describe_dataset.py — описание всех колонок coffee_beans_clean.csv:
тип, заполненность (шт. и %), уникальность, статистика / top-значения.

Запуск:
    python describe_dataset.py --input coffee_beans_clean.csv

Выход:
    dataset_description.csv  — полная таблица
    dataset_description.md   — готовая markdown-таблица для README
    консольный отчёт         — общая сводка + «проблемные» поля
"""
import argparse
import re

import numpy as np
import pandas as pd

# Группировка колонок (совпадает с разделами README)
GROUPS = [
    ("Идентификация и метаданные", [
        "product_url", "product_id", "sku", "scraped_at",
        "parse_status", "http_status", "error_message",
    ]),
    ("Товар", [
        "name", "brand", "product_type", "category_url", "category_name",
    ]),
    ("Продавец", [
        "seller_name", "seller_url", "seller_rating", "seller_orders",
    ]),
    ("Цена и наличие", ["price", "currency", "availability"]),
    ("Рейтинг и отзывы", ["rating", "reviews_count", "rating_count"]),
    ("Вес и упаковка", [
        "weight_g", "packaging", "units_in_pack", "weight_g_actual",
    ]),
    ("Характеристики кофе", [
        "roast_level", "origin_country", "country_of_origin",
        "bean_composition", "arabica_percent", "coffee_processing",
        "flavor_notes", "taste_intensity", "acidity", "bitterness", "strength",
    ]),
    ("Страна-изготовитель и хранение", [
        "manufacturer_country", "shelf_life_days",
        "min_temperature", "max_temperature",
    ]),
    ("Текст и медиа", ["description", "image_url"]),
    ("Сырые данные", ["specs_json"]),
    ("Производные поля очистки", [
        "price_per_kg", "weight_source", "segment",
        "is_outlier_price", "data_quality_flags",
    ]),
]

DERIVED = {
    "units_in_pack", "weight_g_actual",
    "price_per_kg", "weight_source", "segment",
    "is_outlier_price", "data_quality_flags",
}

NULL_STRINGS = {"", "nan", "none", "null", "na", "n/a"}


def is_filled(series: pd.Series) -> pd.Series:
    """True для реально заполненных значений (не NaN, не строка 'nan' и т. п.)."""
    def _ok(v):
        if pd.isna(v):
            return False
        s = str(v).strip()
        return s.lower() not in NULL_STRINGS
    return series.map(_ok)


def fmt_num(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.2f}"


def stats_for_column(s: pd.Series, total: int) -> dict:
    filled_mask = is_filled(s)
    n_filled = int(filled_mask.sum())
    pct = round(100 * n_filled / total, 1) if total else 0.0
    s_f = s[filled_mask]
    n_unique = int(s_f.nunique())

    dtype_str = "str"
    extra = ""

    # пытаемся интерпретировать как число (если ≥ 90% значений парсятся)
    s_num = pd.to_numeric(s_f, errors="coerce")
    n_numeric = int(s_num.notna().sum())
    if n_filled > 0 and n_numeric >= 0.9 * n_filled:
        s_num = s_num.dropna()
        if (s_num == s_num.round()).all():
            dtype_str = "int"
        else:
            dtype_str = "float"
        extra = (
            f"min={fmt_num(s_num.min())}, "
            f"median={fmt_num(s_num.median())}, "
            f"max={fmt_num(s_num.max())}"
        )
    else:
        if n_unique == 0:
            extra = "—"
        elif n_unique <= 20:
            top = s_f.value_counts().head(5)
            extra = "топ: " + ", ".join(f"{v!r} ({c})" for v, c in top.items())
        else:
            extra = f"{n_unique} уник. значений"
            # если строковый — покажем примеры
            sample = s_f.dropna().astype(str).head(2).tolist()
            if sample:
                extra += " | напр.: " + ", ".join(repr(x[:30]) for x in sample)

    return {
        "n_filled": n_filled,
        "pct": pct,
        "n_unique": n_unique,
        "dtype": dtype_str,
        "extra": extra,
    }


def build_markdown(rows: list, total: int, n_cols: int) -> str:
    """Собирает markdown-таблицу по группам. rows — список dict."""
    by_col = {r["column"]: r for r in rows}
    out = []
    out.append(f"**Всего:** {total} строк × {n_cols} колонок.\n")
    out.append(
        "Заполненность — в % от общего числа строк. "
        "Производные поля, добавленные при очистке, помечены `derived`.\n"
    )
    for group_name, cols in GROUPS:
        present = [c for c in cols if c in by_col]
        missing = [c for c in cols if c not in by_col]
        if not present:
            continue
        out.append(f"\n### {group_name}\n")
        out.append("| Колонка | Тип | Заполнено | Уникальных | Статистика / top |")
        out.append("|---|---|---|---|---|")
        for col in present:
            r = by_col[col]
            mark = " `derived`" if col in DERIVED else ""
            fill = f"{r['pct']}% ({r['n_filled']})"
            out.append(
                f"| `{col}`{mark} | {r['dtype']} | {fill} | "
                f"{r['n_unique']} | {r['extra']} |"
            )
        if missing:
            out.append(f"\n_Колонки группы не найдены в датасете: {missing}_\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/coffee_beans_clean.csv")
    ap.add_argument("--out-csv", default="dataset_description.csv")
    ap.add_argument("--out-md", default="dataset_description.md")
    args = ap.parse_args()

    print(f"Читаю {args.input} ...")
    df = pd.read_csv(args.input, dtype=str)
    total = len(df)
    n_cols = len(df.columns)
    print(f"  строк: {total}, колонок: {n_cols}")

    # какие колонки ожидаем увидеть, а какие — лишние
    expected = [c for _, cols in GROUPS for c in cols]
    missing_expected = [c for c in expected if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in expected]
    if missing_expected:
        print(f"\n⚠  Ожидались, но отсутствуют в датасете: {missing_expected}")
    if extra_cols:
        print(f"\n⚠  Есть в датасете, но не описаны в GROUPS: {extra_cols}")

    # описание по каждой колонке
    rows = []
    for group_name, cols in GROUPS:
        for c in cols:
            if c not in df.columns:
                continue
            st = stats_for_column(df[c], total)
            rows.append({"group": group_name, "column": c, **st})
    # лишние колонки — в отдельную группу
    for c in extra_cols:
        st = stats_for_column(df[c], total)
        rows.append({"group": "Прочее", "column": c, **st})

    out_df = pd.DataFrame(rows)[
        ["group", "column", "dtype", "n_filled", "pct", "n_unique", "extra"]
    ]
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"\nТаблица -> {args.out_csv}")

    md = build_markdown(rows, total, n_cols)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Markdown -> {args.out_md}")

    # -------- консольный отчёт --------
    print("\n" + "=" * 78)
    print("ОБЩАЯ СВОДКА")
    print("=" * 78)
    print(f"  строк:    {total}")
    print(f"  колонок:  {n_cols}")

    fully_filled = [r["column"] for r in rows if r["pct"] >= 99.9]
    high = [(r["column"], r["pct"]) for r in rows
            if 90 <= r["pct"] < 99.9]
    mid = [(r["column"], r["pct"]) for r in rows
           if 50 <= r["pct"] < 90]
    low = [(r["column"], r["pct"]) for r in rows
           if 0 < r["pct"] < 50]
    empty = [(r["column"], r["pct"]) for r in rows if r["pct"] == 0]

    print(f"\n  заполнено на 100% ({len(fully_filled)}):")
    print("    " + ", ".join(fully_filled) if fully_filled else "    —")
    if high:
        print(f"\n  заполнено на 90–99% ({len(high)}):")
        for c, p in high:
            print(f"    {c:30s} {p:5.1f}%")
    if mid:
        print(f"\n  заполнено на 50–90% ({len(mid)}):")
        for c, p in mid:
            print(f"    {c:30s} {p:5.1f}%")
    if low:
        print(f"\n  заполнено менее чем на 50% ({len(low)}):")
        for c, p in low:
            print(f"    {c:30s} {p:5.1f}%")
    if empty:
        print(f"\n  ПУСТЫЕ (0%) ({len(empty)}):")
        for c, p in empty:
            print(f"    {c}")

    # -------- поля, которые были "см. скрипт" --------
    watch = [
        "seller_url", "seller_rating", "seller_orders",
        "packaging",
        "origin_country", "country_of_origin",
        "coffee_processing", "flavor_notes",
        "taste_intensity", "acidity", "bitterness", "strength",
        "manufacturer_country",
        "description",
    ]
    print("\n" + "=" * 78)
    print("ПОЛЯ, КОТОРЫЕ РАНЬШЕ БЫЛИ 'см. скрипт' — РЕАЛЬНЫЕ ЗНАЧЕНИЯ")
    print("=" * 78)
    print(f"  {'колонка':25s} {'%':>7s} {'n':>6s}  {'уник':>6s}")
    for c in watch:
        r = next((x for x in rows if x["column"] == c), None)
        if r is None:
            print(f"  {c:25s}  НЕ НАЙДЕНА В ДАТАСЕТЕ")
            continue
        print(
            f"  {c:25s} {r['pct']:6.1f}% {r['n_filled']:6d}  "
            f"{r['n_unique']:6d}"
        )

    print("\n" + "=" * 78)
    print(f"Готово. Скопируйте {args.out_md} в README → раздел «Структура датасета».")
    print("=" * 78)


if __name__ == "__main__":
    main()