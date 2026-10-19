import argparse
import asyncio
import csv
import os
import re
from typing import Optional

from playwright.async_api import async_playwright

CATEGORY_URL = "https://www.ozon.ru/category/kofe-v-zernah-31009/"
DEFAULT_OUT = "data/raw/coffee_beans_category_urls.csv"
DEFAULT_MAX_SCROLLS = 2000
SAVE_EVERY = 10
CDP_URL = "http://localhost:9222"

SKU_RE = re.compile(r"/product/[^/?#]+-(\d+)/?$")


def normalize_product_url(url: str) -> Optional[str]:
    if "/product/" not in url:
        return None
    base = url.split("?", 1)[0].split("#", 1)[0]
    if not SKU_RE.search(base):
        return None
    if not base.endswith("/"):
        base += "/"
    return base


def extract_sku(url: str) -> Optional[str]:
    m = SKU_RE.search(url)
    return m.group(1) if m else None


def load_existing(path: str) -> set:
    """Читает существующий CSV и возвращает множество URL."""
    if not os.path.exists(path):
        return set()
    links = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = (row.get("product_url") or "").strip()
            if url:
                n = normalize_product_url(url)
                if n:
                    links.add(n)
    return links


def save_csv(path: str, links: set):
    """Атомарная запись: сначала в .tmp, потом переименование."""
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["product_url", "sku"])
        skus = set()
        for u in sorted(links):
            sku = extract_sku(u)
            if sku:
                skus.add(sku)
            w.writerow([u, sku or ""])
    os.replace(tmp, path)
    return len(skus)


async def get_product_links(page) -> set:
    urls = await page.eval_on_selector_all(
        'a[href*="/product/"]',
        "els => els.map(a => a.href)",
    )
    result = set()
    for u in urls:
        n = normalize_product_url(u)
        if n:
            result.add(n)
    return result


async def scroll_down(page):
    await page.evaluate("""
        () => {
            const el = document.scrollingElement || document.documentElement;
            el.scrollTop = el.scrollHeight;
        }
    """)


async def get_ozon_page(browser):
    if not browser.contexts:
        ctx = await browser.new_context()
    else:
        ctx = browser.contexts[0]
    for p in ctx.pages:
        if "ozon.ru" in p.url:
            return p
    return await ctx.new_page()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-scrolls", type=int, default=DEFAULT_MAX_SCROLLS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--category", default=CATEGORY_URL)
    parser.add_argument("--no-resume", action="store_true",
                        help="не подхватывать существующий CSV, начать с нуля")
    args = parser.parse_args()

    # 1. Загружаем то, что уже собрано (если есть)
    if args.no_resume:
        all_links = set()
        print(f"--no-resume: начинаем с нуля, файл {args.out} будет перезаписан")
    else:
        all_links = load_existing(args.out)
        if all_links:
            print(f"Подхватили из {args.out}: {len(all_links)} ссылок")
        else:
            print(f"Файл {args.out} не найден или пуст — начинаем с нуля")

    # 2. Коннект и подготовка страницы
    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(CDP_URL, timeout=30000)
        page = await get_ozon_page(browser)
        await page.bring_to_front()

        try:
            await page.goto(args.category, wait_until="domcontentloaded", timeout=60000)
        except Exception as e:
            print(f"Не удалось открыть категорию: {e}")
            return

        await page.wait_for_timeout(3000)

        stable_rounds = 0
        start_count = len(all_links)

        for i in range(args.max_scrolls):
            links = await get_product_links(page)
            new_links = links - all_links
            all_links.update(links)

            print(
                f"[скролл {i + 1}] в DOM: {len(links)}, "
                f"новых: {len(new_links)}, всего: {len(all_links)}"
            )

            if (i + 1) % SAVE_EVERY == 0:
                n = save_csv(args.out, all_links)
                print(f"  ↳ сохранили: {len(all_links)} ссылок, {n} SKU")

            # Остановка — только если мы уже видели какие-то товары вообще
            # (не считаем «новым» то, что загрузили из CSV) и лента залипла.
            if len(new_links) == 0:
                stable_rounds += 1
                if stable_rounds >= 5:
                    print("5 раундов без новых ссылок — останавливаемся.")
                    break
            else:
                stable_rounds = 0

            await scroll_down(page)
            await page.wait_for_timeout(1500)

        skus = save_csv(args.out, all_links)
        added = len(all_links) - start_count
        print(f"\nГотово: {len(all_links)} уникальных ссылок "
              f"(+{added} за этот запуск), "
              f"{skus} уникальных SKU -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())