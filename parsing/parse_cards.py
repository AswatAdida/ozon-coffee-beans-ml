import argparse
import asyncio
import json
import os
import random
import re
from datetime import datetime

import pandas as pd
from playwright.async_api import async_playwright

CHECKPOINT_EVERY = 10
DEFAULT_DELAY = 1.5
CONSECUTIVE_FAIL_LIMIT = 5

CATEGORY_URL = "https://www.ozon.ru/category/kofe-v-zernah-10519/"
CATEGORY_NAME = "Кофе в зернах"

SPEC_ALIASES = {
    "product_type":         ["тип"],
    "roast_level":          ["степень обжарки", "обжарка"],
    "origin_country":       ["страна происхождения зерен кофе", "страна происхождения зёрен кофе", "страна происхождения"],
    "bean_composition":     ["состав кофе", "состав"],
    "arabica_percent":      ["содержание арабики, %", "содержание арабики"],
    "taste_intensity":      ["интенсивность вкуса"],
    "manufacturer_country": ["страна-изготовитель", "страна изготовитель"],
    "shelf_life_days":      ["срок годности в днях", "срок годности"],
    "min_temperature":      ["минимальная температура"],
    "max_temperature":      ["максимальная температура"],
    "weight_g":             ["вес товара, г", "вес товара"],
    "packaging":            ["упаковка", "тип упаковки"],
    "flavor_notes":         ["вкусовые ноты", "ноты вкуса"],
    "coffee_processing":    ["способ обработки", "обработка"],
    "acidity":              ["кислинка", "кислотность"],
    "bitterness":           ["горчинка", "горечь"],
    "strength":             ["крепость"],
}

COLUMNS = [
    "product_url", "product_id", "sku", "scraped_at",
    "parse_status", "http_status", "error_message",
    "name", "brand", "product_type", "category_url", "category_name",
    "seller_name", "seller_url", "seller_rating", "seller_orders",
    "price", "currency", "availability",
    "rating", "reviews_count", "rating_count",
    "weight_g", "packaging",
    "roast_level", "origin_country", "country_of_origin",
    "bean_composition", "arabica_percent", "coffee_processing",
    "flavor_notes", "taste_intensity", "acidity", "bitterness", "strength",
    "manufacturer_country", "shelf_life_days", "min_temperature", "max_temperature",
    "description", "image_url", "specs_json",
]


def normalize_label(v):
    if v is None:
        return ""
    v = str(v).lower().replace("ё", "е")
    return re.sub(r"\s+", " ", v).strip(" :\t\r\n")


def walk_json(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk_json(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_json(v)


def find_products(scripts):
    result = []
    for raw in scripts:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        for obj in walk_json(data):
            t = obj.get("@type", [])
            if isinstance(t, str):
                t = [t]
            if any(str(x).lower() == "product" for x in t):
                result.append(obj)
    return result


def pick_best_product(products):
    if not products:
        return None
    def score(p):
        return sum(1 for k in ("name", "brand", "offers", "aggregateRating",
                               "description", "sku", "image") if p.get(k))
    return max(products, key=score)


def extract_brand(b):
    if isinstance(b, dict):
        return b.get("name") or b.get("@id")
    if isinstance(b, list) and b:
        return extract_brand(b[0])
    return b


def parse_float(v):
    if v is None:
        return None
    try:
        return float(str(v).replace(",", ".").strip())
    except (ValueError, TypeError):
        return None


def parse_int(v):
    if v is None:
        return None
    s = re.sub(r"[^\d\-]", "", str(v))
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def normalize_weight_g(value, name):
    if value:
        f = parse_float(str(value).replace(",", "."))
        if f:
            return round(f)
    if name:
        text = str(name).lower().replace(",", ".")
        m = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*(кг|kg)\b", text)
        if m:
            return round(float(m.group(1)) * 1000)
        m = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*(г|гр|g)\b", text)
        if m:
            return round(float(m.group(1)))
    return None


def normalize_bean(raw):
    if not raw:
        return None, None
    text = str(raw).strip()
    low = text.lower().replace("ё", "е")
    percent = None
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*%\s*арабик", low)
    if m:
        percent = int(round(float(m.group(1).replace(",", "."))))
    has_ar = "арабик" in low
    has_rob = "робуст" in low
    if has_ar and has_rob:
        return "Арабика/Робуста", percent
    if has_ar:
        return "Арабика", percent
    if has_rob:
        return "Робуста", percent
    return text, None


def normalize_arabica(raw, bean, bean_pct):
    if raw:
        v = parse_int(raw)
        if v is not None and 0 <= v <= 100:
            return v
    if bean_pct is not None:
        return bean_pct
    if bean == "Арабика":
        return 100
    return None


def build_specs_json(pairs):
    result = {}
    for name, value in pairs:
        if not name or value is None:
            continue
        if name in result:
            if isinstance(result[name], list):
                result[name].append(value)
            else:
                result[name] = [result[name], value]
        else:
            result[name] = value
    return json.dumps(result, ensure_ascii=False)


async def parse_card(page, url, sku):
    row = {c: None for c in COLUMNS}
    row["product_url"] = url
    row["sku"] = sku or None
    row["scraped_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row["parse_status"] = "failed"
    row["category_url"] = CATEGORY_URL
    row["category_name"] = CATEGORY_NAME

    try:
        resp = await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        row["http_status"] = resp.status if resp else 200

        await page.wait_for_timeout(random.randint(2000, 4000))

        if resp and resp.status >= 400:
            row["error_message"] = f"http_{resp.status}"
            return row

        scripts = await page.locator('script[type="application/ld+json"]').all_text_contents()
        product = pick_best_product(find_products(scripts))
        if product:
            n = product.get("name")
            row["name"] = str(n).strip() if n else None

            b = extract_brand(product.get("brand"))
            row["brand"] = str(b).strip() if b else None

            s = product.get("sku")
            if s:
                row["product_id"] = str(s).strip()

            offers = product.get("offers")
            if isinstance(offers, list):
                offers = offers[0] if offers else {}
            if isinstance(offers, dict):
                p = offers.get("price") if offers.get("price") is not None else offers.get("lowPrice")
                if p is not None:
                    row["price"] = str(p)
                row["currency"] = offers.get("priceCurrency")
                av = offers.get("availability")
                if av:
                    row["availability"] = str(av).split("/")[-1]

            agg = product.get("aggregateRating") or {}
            if isinstance(agg, dict):
                if agg.get("ratingValue") is not None:
                    row["rating"] = str(agg["ratingValue"])
                if agg.get("reviewCount") is not None:
                    row["reviews_count"] = str(agg["reviewCount"])
                if agg.get("ratingCount") is not None:
                    row["rating_count"] = str(agg["ratingCount"])

            d = product.get("description")
            if d:
                row["description"] = str(d).strip()

            img = product.get("image")
            if isinstance(img, list) and img:
                img = img[0]
            if isinstance(img, dict):
                img = img.get("url")
            if img:
                row["image_url"] = str(img)

        async def _extract_specs():
            try:
                await page.wait_for_selector("#section-characteristics dl", timeout=8000)
                return await page.evaluate("""() => {
                    const out = [];
                    document.querySelectorAll('#section-characteristics dl').forEach(dl => {
                        const dt = dl.querySelector('dt');
                        const dd = dl.querySelector('dd');
                        if (!dt || !dd) return;
                        const name = (dt.textContent || '').trim();
                        const val = (dd.textContent || '').trim();
                        if (name) out.push([name, val]);
                    });
                    return out;
                }""")
            except Exception:
                return []

        pairs = await _extract_specs()
        # Retry: если характеристик нет — прокрутить вниз, подождать, попробовать снова
        if not pairs:
            try:
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                await page.wait_for_timeout(2500)
                await page.evaluate("window.scrollTo(0, 0)")
                await page.wait_for_timeout(1000)
            except Exception:
                pass
            pairs = await _extract_specs()

        row["specs_json"] = build_specs_json(pairs) if pairs else None

        spec_map = {}
        for name, val in pairs:
            key = normalize_label(name)
            if key and key not in spec_map:
                spec_map[key] = val

        for col, aliases in SPEC_ALIASES.items():
            for a in aliases:
                k = normalize_label(a)
                if k in spec_map:
                    row[col] = spec_map[k]
                    break
                # Fallback: если JSON-LD не дал brand — берём из характеристик
        if not row.get("brand"):
            brand_spec = spec_map.get(normalize_label("Бренд"))
            if brand_spec:
                row["brand"] = brand_spec.strip()

        bean_raw = row.get("bean_composition")
        bean_norm, bean_pct = normalize_bean(bean_raw)
        row["bean_composition"] = bean_norm
        row["arabica_percent"] = normalize_arabica(row.get("arabica_percent"), bean_norm, bean_pct)

        row["weight_g"] = normalize_weight_g(row.get("weight_g"), row.get("name"))

        if row.get("shelf_life_days"):
            row["shelf_life_days"] = parse_int(row["shelf_life_days"])

        for fld in ("min_temperature", "max_temperature"):
            if row.get(fld):
                row[fld] = parse_float(row[fld])

        row["country_of_origin"] = row.get("origin_country")

                # --- Продавец (устойчивые селекторы с fallback) ---
        async def _extract_seller():
            return await page.evaluate(r"""() => {
                const r = {name: null, url: null, rating: null, orders: null};

                const links = [...document.querySelectorAll('a[href*="/seller/"]')];
                const withText = links.find(a => (a.innerText || '').trim());
                if (withText) r.url = withText.href;
                else if (links.length) r.url = links[0].href;

                const nameEls = document.querySelectorAll('.b35_5_4-b7');
                for (const el of nameEls) {
                    const t = (el.innerText || '').trim();
                    if (t) { r.name = t; break; }
                }

                const allDivs = [...document.querySelectorAll('div')];
                const shop = allDivs.find(el => {
                    const t = el.innerText || '';
                    return t.includes('Магазин') && t.includes('Заказы') 
                           && el.children.length < 30 && t.length < 800;
                });

                if (shop) {
                    const ratingEls = shop.querySelectorAll('.tsCompactControl300XSmall');
                    for (const el of ratingEls) {
                        const t = (el.innerText || '').trim();
                        if (/^\d+[.,]\d+$/.test(t)) { r.rating = t; break; }
                    }
                }

                if (!r.rating) {
                    const all = document.querySelectorAll('.tsCompactControl300XSmall');
                    for (const el of all) {
                        const t = (el.innerText || '').trim();
                        if (/^\d+[.,]\d+$/.test(t)) { r.rating = t; break; }
                    }
                }

                const orderLabels = [...document.querySelectorAll('span, div')].filter(el => 
                    (el.innerText || '').trim().toLowerCase() === 'заказы' 
                    && el.children.length === 0
                );

                for (const label of orderLabels) {
                    let p = label.parentElement;
                    for (let i = 0; i < 3 && p; i++) {
                        const titleEl = p.querySelector('[title]');
                        if (titleEl) {
                            const title = (titleEl.getAttribute('title') || '').trim();
                            if (/^\d[\d\s.,]*\s*[KkMmКкМм]?$/.test(title) && title.length < 15) {
                                r.orders = title;
                                break;
                            }
                        }
                        const siblingText = (p.innerText || '')
                            .replace(/^.*?Заказы\s*/i, '')
                            .split('\n')[0]
                            .trim();
                        if (/^\d[\d\s.,]*\s*[KkMmКкМм]?$/.test(siblingText) 
                            && siblingText.length < 15) {
                            r.orders = siblingText;
                            break;
                        }
                        p = p.parentElement;
                    }
                    if (r.orders) break;
                }

                return r;
            }""")

        try:
            seller = await _extract_seller()
            # Если имя продавца не нашли — возможно, страница не догрузилась. Ждём и пробуем ещё раз.
            if not seller or not seller.get("name"):
                await page.wait_for_timeout(2500)
                seller = await _extract_seller()

            if seller:
                if seller.get("name"):   row["seller_name"]   = seller["name"]
                if seller.get("url"):    row["seller_url"]    = seller["url"]
                if seller.get("rating"): row["seller_rating"] = seller["rating"]
                if seller.get("orders"): row["seller_orders"] = seller["orders"]
        except Exception as e:
            row["error_message"] = (row.get("error_message") or "") + f" seller_err:{str(e)[:80]}"

        key_fields = ["name", "brand", "price", "rating", "reviews_count",
                      "seller_name", "roast_level", "bean_composition"]
        filled = sum(1 for f in key_fields if row.get(f))
        if filled >= 5:
            row["parse_status"] = "ok"
        elif filled >= 3:
            row["parse_status"] = "ok_partial"
        else:
            # Товар удалён/битая ссылка — это не проблема соединения.
            # Помечаем отдельно, чтобы не считать в consecutive_fails
            # и больше не возвращаться к этой карточке.
            row["parse_status"] = "not_available"
            row["error_message"] = f"only_{filled}_key_fields"

        return row

    except Exception as e:
        row["parse_status"] = "failed"
        row["error_message"] = str(e)[:300]
        return row


def save(rows, path):
    if not rows:
        return
    df = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[COLUMNS]
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/coffee_beans_category_urls.csv")
    ap.add_argument("--out", default="data/raw/coffee_beans_parsed.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Не найден {args.input}")
        return

    df_in = pd.read_csv(args.input, dtype=str)
    if "product_url" not in df_in.columns:
        print("В input нет колонки product_url")
        return
    if "sku" not in df_in.columns:
        df_in["sku"] = df_in["product_url"].str.extract(r"(\d+)/?$")

    done_skus = set()
    existing = []
    if not args.no_resume and os.path.exists(args.out):
        try:
            prev = pd.read_csv(args.out, dtype=str)
            ok = prev[prev["parse_status"].isin(["ok", "ok_partial", "not_available"])]
            done_skus = set(ok["sku"].dropna().astype(str))
            existing = prev.to_dict("records")
            print(f"Resume: {args.out} -> успешных {len(done_skus)}, всего строк {len(prev)}")
        except Exception as e:
            print(f"Не удалось прочитать {args.out}: {e}")

    pending = []
    for _, r in df_in.iterrows():
        sku = str(r.get("sku") or "").strip()
        url = str(r.get("product_url") or "").strip()
        if not url:
            continue
        if sku and sku in done_skus:
            continue
        pending.append({"sku": sku, "url": url})

    if args.limit:
        pending = pending[:args.limit]

    print(f"К обработке: {len(pending)} карточек")
    if not pending:
        print("Нечего делать.")
        return

    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp("http://localhost:9222")
        if not browser.contexts:
            print("Нет доступного контекста Chrome")
            return
        context = browser.contexts[0]
        page = await context.new_page()

        consec_fails = 0
        try:
            for i, item in enumerate(pending, 1):
                sku, url = item["sku"], item["url"]
                row = await parse_card(page, url, sku)
                existing.append(row)

                print(f"[{i}/{len(pending)}] {sku} | {row['parse_status']} | "
                      f"name={str(row.get('name'))[:40]!r} | "
                      f"seller={row.get('seller_name')!r} | "
                      f"rating={row.get('seller_rating')} | "
                      f"orders={row.get('seller_orders')} | "
                      f"reviews={row.get('reviews_count')} | "
                      f"err={row.get('error_message')}")

                if row["parse_status"] == "failed":
                    consec_fails += 1
                else:
                    consec_fails = 0

                if consec_fails >= CONSECUTIVE_FAIL_LIMIT:
                    print(f"\n!!! {consec_fails} failed подряд — возможно, капча или VPN.")
                    print("!!! Зайди в Chrome, проверь соединение, перезапусти скрипт.\n")
                    save(existing, args.out)
                    return

                if i % CHECKPOINT_EVERY == 0:
                    save(existing, args.out)
                    print(f"--- checkpoint: {i}/{len(pending)} ---")

                await asyncio.sleep(args.delay + random.uniform(0, 1.5))
        finally:
            save(existing, args.out)
            await page.close()

    print(f"\nГотово: {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
