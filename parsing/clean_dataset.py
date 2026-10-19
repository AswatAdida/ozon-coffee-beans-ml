"""
clean_dataset.py — очистка датасета кофе в зёрнах.

Принцип: НЕ удаляем, а ПРАВИМ грязные поля.
Удаляем только то, что вообще не кофе в зёрнах (молотый, растворимый,
капсулы, дрип, 3-в-1), снятые с продажи и не-RUB.

Что правим:
  - units_in_pack: если в названии нет мультипака, а в спеке >1 — это баг Ozon
    (типа "Единиц в одном товаре = 10" на 1 кг). Ставим 1.
  - weight_g_actual: для мультипаков ("3 шт по 250 г", "2 пачки по 1 кг",
    "2x250 g") берём вес из названия = N × M, а не умножаем спеку на units.
  - брутто-вес ("1002", "1016", "1037") округляем до ближайшего "круглого",
    если отклонение < 15%.
  - price_per_kg пересчитываем заново от исправленных weight/units.

Что НЕ удаляем, но помечаем:
  - bulk (> 10 кг) → segment='bulk'
  - sample (< 100 г) → segment='sample'
  - price_per_kg вне 200–80000 или в "серой зоне" 200–300 / 60000–80000
    → is_outlier_price=True (для ML можно потом отфильтровать)
"""

import json
import re
from collections import Counter

import pandas as pd

INPUT = "data/raw/coffee_beans_parsed.csv"
OUTPUT = "data/processed/coffee_beans_clean.csv"

BRAND_FIX = {
    "gemma": "Gemma", "rockets": "Rockets", "bushido": "Bushido",
    "movenpick": "Movenpick", "julius meinl": "Julius Meinl",
    "egoiste": "EGOISTE", "lavazza": "Lavazza", "vnc": "VNC",
    "kurukahveci mehmet efendi": "Kurukahveci Mehmet Efendi",
}

# Допустимые "круглые" веса кофе (г).
ROUND_WEIGHTS = {
    50, 100, 150, 200, 227, 250, 300, 340, 350, 375, 400, 450, 500,
    600, 700, 750, 800, 900, 907, 1000, 1500, 2000, 2270, 2500,
    3000, 4000, 5000, 10000,
}

# ---------- regex для парсинга названия ----------
# Длинные единицы — вперёд, чтобы "грамм" не обрезалось до "г".
# БЕЗ \b: "250гх2шт" — после "г" идёт "х" (обе кириллические), \b не сработает.
_WEIGHT_UNITS = r"(грамм[а-я]*|килограмм[а-я]*|кг|kg|гр|gr|г|g)"
_WEIGHT_UNITS_B = _WEIGHT_UNITS + r"\b"  # с boundary — для forward-мультипака

_WEIGHT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*" + _WEIGHT_UNITS,
    re.IGNORECASE,
)
_COUNT_RE = re.compile(
    r"(\d+)\s*(?:шт\w*|пачк\w*|упаков\w*|уп)\b", re.IGNORECASE
)
# "N по M unit" / "N × M unit" / "3 шт по 250 г"
_MULTIPACK_RE = re.compile(
    r"(\d+)\s*(?:шт\w*|пачк\w*|упаков\w*|уп)?\s*"
    r"(?:по|[xх×*])\s*"
    r"(\d+(?:[.,]\d+)?)\s*" + _WEIGHT_UNITS_B,
    re.IGNORECASE,
)
# "M unit × N шт" / "250гх2шт" — обратный порядок
_MULTIPACK_RE_REV = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*" + _WEIGHT_UNITS + r"\s*"
    r"[xх×*]\s*(\d+)\s*(?:шт\w*|уп\w*|пачк\w*)?",
    re.IGNORECASE,
)
# "N кг (N шт)" / "10 кг (10 шт.)" — N кг это СУММАРНЫЙ вес, N шт — фасовка
_SAME_N_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*" + _WEIGHT_UNITS + r"\s*[\(\[]?\s*"
    r"\1\s*(?:шт\w*|уп\w*|пачк\w*|комплект\w*)\b",
    re.IGNORECASE,
)


# ---------- безопасные конвертеры ----------
def to_float(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (ValueError, TypeError):
        pass
    if isinstance(v, str):
        s = v.replace(",", ".").replace(" ", "").strip()
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


def to_int(v):
    f = to_float(v)
    return int(round(f)) if f is not None else None


# ---------- units из specs_json (с защитой от list) ----------
def extract_units(specs_json):
    """Возвращает (units, source) — units из specs, source нужен для флагов."""
    if not specs_json or pd.isna(specs_json):
        return 1, None
    try:
        s = json.loads(specs_json)
        if not isinstance(s, dict):
            return 1, None
        for key in ("Единиц в одном товаре", "Количество в упаковке, шт"):
            if key in s:
                raw = s[key]
                # защита от list-бага (дубликат ключа)
                if isinstance(raw, list):
                    raw = raw[0] if raw else None
                if raw is None:
                    continue
                digits = re.sub(r"[^\d]", "", str(raw))
                if digits:
                    return int(digits), key
    except Exception:
        pass
    return 1, None


# ---------- парсинг названия ----------
def parse_name(name):
    """
    Возвращает dict:
      total_g:     суммарный вес из названия (N×M или M)
      units:       количество единиц из названия (1, если не найдено)
      per_unit_g:  вес одной единицы
      is_mp:       True, если в названии явный мультипак
    """
    empty = {"total_g": None, "units": 1, "per_unit_g": None, "is_mp": False}
    if not name or not isinstance(name, str):
        return empty

    text = name.lower().replace(",", ".")

    def _to_g(val, unit):
        if unit in ("кг", "kg") or unit.startswith("килограмм"):
            return val * 1000
        return val

    # 0) "N кг (N шт)" / "10 кг (10 шт.)" — N кг = СУММАРНЫЙ вес
    m = _SAME_N_RE.search(text)
    if m:
        n_val = float(m.group(1))
        unit = m.group(2).lower()
        total = _to_g(n_val, unit)
        n = max(1, int(round(n_val)))
        return {
            "total_g": round(total),
            "units": n,
            "per_unit_g": round(total / n) if n else None,
            "is_mp": n > 1,
        }

    # 1) "N по M unit" / "N × M unit"
    m = _MULTIPACK_RE.search(text)
    if m:
        n = int(m.group(1))
        val = float(m.group(2))
        unit = m.group(3).lower()
        per_unit = _to_g(val, unit)
        if per_unit > 0:
            return {
                "total_g": round(n * per_unit),
                "units": n,
                "per_unit_g": round(per_unit),
                "is_mp": n > 1,
            }

    # 2) "M unit × N шт" / "250гх2шт"
    m = _MULTIPACK_RE_REV.search(text)
    if m:
        val = float(m.group(1))
        unit = m.group(2).lower()
        n = int(m.group(3))
        per_unit = _to_g(val, unit)
        if per_unit > 0:
            return {
                "total_g": round(n * per_unit),
                "units": n,
                "per_unit_g": round(per_unit),
                "is_mp": n > 1,
            }

    # 3) fallback: "N штук" + вес
    counts = [int(x) for x in _COUNT_RE.findall(text)]
    n_units = max(counts) if counts else 1

    weights = []
    for m in _WEIGHT_RE.finditer(text):
        val = float(m.group(1))
        unit = m.group(2).lower()
        val = _to_g(val, unit)
        if val > 0:
            weights.append(val)
    if not weights:
        return {"total_g": None, "units": n_units,
                "per_unit_g": None, "is_mp": n_units > 1}

    per_unit = max(weights)
    return {
        "total_g": round(n_units * per_unit) if n_units > 1 else round(per_unit),
        "units": n_units,
        "per_unit_g": round(per_unit),
        "is_mp": n_units > 1,
    }


def nearest_round(w, tol=0.15):
    """Ближайший "круглый" вес, если отклонение < tol; иначе исходный."""
    if w is None or w <= 0:
        return w
    best, best_diff = None, float("inf")
    for r in ROUND_WEIGHTS:
        d = abs(w - r) / r
        if d < best_diff:
            best_diff, best = d, r
    return best if best_diff <= tol else w


def fix_brand(b):
    if b is None or (isinstance(b, float) and pd.isna(b)):
        return None
    low = str(b).strip().lower()
    return BRAND_FIX.get(low, str(b).strip())


# ---------- главная функция: разрешение веса ----------
def resolve_weight(row):
    """
    Возвращает:
      weight_actual, units_final, source, flags(list), segment
    source ∈ {specs, name_single, name_multipack, none}
    """
    flags = []
    name = row.get("name")
    wg = to_float(row.get("weight_g"))
    units_specs, units_source = extract_units(row.get("specs_json"))
    nm = parse_name(name)
    w_name = nm["total_g"]
    units_name = nm["units"]
    per_unit_name = nm["per_unit_g"]
    is_mp_name = nm["is_mp"]

    weight_actual = None
    units_final = 1
    source = "specs"

    def _close(a, b, tol=0.15):
        if a is None or b is None or b == 0:
            return False
        return abs(a - b) / max(b, 1) < tol

    # ---------- Кейс 1: мультипак в названии ----------
    if is_mp_name and units_name > 1:
        if units_specs > 1 and units_specs == units_name:
            # units согласованы, разбираемся с весом
            if wg is None:
                weight_actual, source = w_name, "name_multipack"
                flags.append("weight_from_name")
            elif _close(wg, w_name):
                # weight_g уже суммарный, НЕ умножаем
                weight_actual = w_name
                units_final = 1
                source = "name_multipack"
                flags.append("units_normalized")
            elif per_unit_name and _close(wg, per_unit_name):
                # weight_g за пачку, надо × units
                weight_actual = w_name
                units_final = units_specs
                source = "name_multipack"
                flags.append("weight_multiplied")
            else:
                weight_actual = w_name
                units_final = units_name
                source = "name_multipack"
                flags.append("weight_from_name")
        elif units_specs > 1 and units_specs != units_name:
            # units расходятся — доверяем названию
            weight_actual = w_name
            units_final = units_name
            source = "name_multipack"
            flags.append("units_from_name")
        else:
            # units_specs == 1, но в названии мультипак
            if wg is not None and _close(wg, w_name):
                weight_actual, units_final, source = wg, 1, "specs"
            elif wg is not None and per_unit_name and _close(wg, per_unit_name):
                weight_actual, units_final = w_name, units_name
                source = "name_multipack"
                flags.append("weight_multiplied")
            else:
                weight_actual, units_final = w_name, units_name
                source = "name_multipack"
                flags.append("weight_from_name")

    # ---------- Кейс 2: units>1 в спеке, но в названии мультипака нет ----------
    elif units_specs > 1:
        # баг Ozon: "Единиц в одном товаре=10" на 1 кг
        weight_actual = wg if wg is not None else w_name
        units_final = 1
        source = "specs"
        flags.append("units_normalized")
        if units_source:
            flags.append(f"units_from_{units_source}")

    # ---------- Кейс 3: обычный товар ----------
    else:
        if wg is not None:
            if w_name is not None and _close(wg, w_name):
                rounded = nearest_round(wg)
                weight_actual = rounded
                source = "specs"
                if rounded != wg:
                    flags.append("weight_rounded")
            elif w_name is not None:
                # сильное расхождение
                # если spec weight > 1.5× от name и name — "круглый" вес,
                # то spec похож на брутто — доверяем названию
                if wg > w_name * 1.5 and int(round(w_name)) in ROUND_WEIGHTS:
                    weight_actual = round(w_name)
                    source = "name_single"
                    flags.append("weight_from_name_preferred")
                else:
                    rounded = nearest_round(wg, tol=0.20)
                    weight_actual = rounded
                    source = "specs"
                    flags.append("weight_name_mismatch")
                    if rounded != wg:
                        flags.append("weight_rounded")
            else:
                rounded = nearest_round(wg)
                weight_actual = rounded
                source = "specs"
                if rounded != wg:
                    flags.append("weight_rounded")
        else:
            if w_name is not None:
                weight_actual, source = w_name, "name_single"
                flags.append("weight_from_name")
            else:
                weight_actual, source = None, "none"
                flags.append("weight_null")

    # ---------- сегмент ----------
    segment = "consumer"
    if weight_actual is not None:
        if weight_actual > 10000:
            segment = "bulk"
        elif weight_actual < 100:
            segment = "sample"

    return weight_actual, units_final, source, flags, segment


# ---------- keyword-фильтр: это вообще не зерно? ----------
_GROUND_RE = re.compile(r"\bмолот(ый|ого|ому|ым|ом|ая|ой|ую|ое|ые|ых)\b", re.IGNORECASE)
_INSTANT_RE = re.compile(r"растворим", re.IGNORECASE)
_CAPSULE_RE = re.compile(r"капсул", re.IGNORECASE)
_DRIP_RE = re.compile(r"\bдрип", re.IGNORECASE)
_THREE_IN_ONE_RE = re.compile(r"\d\s*в\s*1\b", re.IGNORECASE)


def not_beans_reason(row):
    """Возвращает причину, почему строку надо удалить, или None."""
    name = row.get("name")
    if isinstance(name, str):
        if _GROUND_RE.search(name):
            return "ground"
        if _INSTANT_RE.search(name):
            return "instant"
        if _CAPSULE_RE.search(name):
            return "capsule"
        if _DRIP_RE.search(name):
            return "drip"
        if _THREE_IN_ONE_RE.search(name):
            return "three_in_one"
    return None


# ---------- price_per_kg ----------
def price_per_kg(price, weight_g):
    try:
        p = float(price)
        w = float(weight_g)
    except (TypeError, ValueError):
        return None
    if pd.isna(p) or pd.isna(w) or w == 0:
        return None
    return round(p / w * 1000)


def main():
    print(f"Читаю {INPUT} ...")
    df = pd.read_csv(INPUT, dtype=str)
    print(f"  строк: {len(df)}")

    # 1. базовые фильтры (это не кофе в зёрнах / снято / не RUB)
    n0 = len(df)
    df = df[df["product_type"] == "Кофе в зернах"].copy()
    df = df[df["parse_status"] != "not_available"].copy()
    df = df[df["currency"].fillna("RUB").str.upper() == "RUB"].copy()

    reasons = df.apply(not_beans_reason, axis=1)
    df = df[reasons.isna()].copy()
    print(f"После фильтра типа/статуса/не-зерна: {len(df)} (удалено {n0 - len(df)})")

    # 2. нормализация бренда
    df["brand"] = df["brand"].apply(fix_brand)

    # 3. разрешение веса и units
    resolved = df.apply(resolve_weight, axis=1)
    df["weight_g_actual"] = [r[0] for r in resolved]
    df["units_in_pack"] = [r[1] for r in resolved]
    df["weight_source"] = [r[2] for r in resolved]
    df["_flags"] = [r[3] for r in resolved]
    df["segment"] = [r[4] for r in resolved]

    # 4. пересчёт цены за кг
    df["price_parsed"] = df["price"].apply(to_float)
    df["price_per_kg"] = [
        price_per_kg(p, w) for p, w in zip(df["price_parsed"], df["weight_g_actual"])
    ]

    # 5. флаг выброса (не удаляем, только помечаем)
    def _is_outlier(x):
        if x is None or pd.isna(x):
            return False
        return x <= 200 or x >= 80000 or (200 < x < 300) or (60000 < x < 80000)
    df["is_outlier_price"] = df["price_per_kg"].apply(_is_outlier)

    # 6. единая колонка флагов
    df["data_quality_flags"] = df["_flags"].apply(lambda f: ",".join(f) if f else "")
    df = df.drop(columns=["_flags", "price_parsed"])

    # 7. сохранение
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"\nСохранено: {OUTPUT} ({len(df)} строк)")

    # ---------- отчёт ----------
    print("\n=== Исправления (data_quality_flags) ===")
    flag_counter = Counter()
    for flags in df["data_quality_flags"]:
        if flags:
            for f in flags.split(","):
                flag_counter[f] += 1
    for flag, cnt in flag_counter.most_common():
        print(f"  {flag:30s} {cnt:5d}")

    print("\n=== weight_source ===")
    print(df["weight_source"].value_counts().to_string())

    print("\n=== segment ===")
    print(df["segment"].value_counts().to_string())

    print("\n=== price_per_kg ===")
    print(f"  не пустых: {df['price_per_kg'].notna().sum()}")
    print(f"  выбросов (is_outlier_price): {df['is_outlier_price'].sum()}")

    print("\n=== Top-5 по price_per_kg (после правок) ===")
    print(
        df.nlargest(5, "price_per_kg")[
            ["sku", "name", "price", "weight_g_actual", "units_in_pack", "price_per_kg"]
        ].to_string(index=False)
    )
    print("\n=== Top-5 по абсолютной price ===")
    df_num = df.assign(_price_num=df["price"].apply(to_float))
    print(
        df_num.nlargest(5, "_price_num")[
            ["sku", "name", "price", "weight_g_actual", "units_in_pack", "price_per_kg"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()