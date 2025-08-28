#!/usr/bin/env python3
import csv, os, random, time
from pathlib import Path
from dataclasses import dataclass, asdict

import pandas as pd
from bs4 import BeautifulSoup

# --- use undetected chromedriver ---
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# ---------- Config ----------
OUT_CSV = Path("imdb_2024_movies_storylines.csv")
PROFILE_DIR = os.path.expanduser("~/imdb-scrape-profile")   # persistent cookie/consent
UA_LIST = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

BASE_URL = (
    "https://www.imdb.com/search/title/"
    "?title_type=feature"
    "&release_date=2024-01-01,2024-12-31"
    "&view=advanced"
    "&sort=moviemeter,asc"
    "&count=50"   # stay conservative to avoid rate limits
)

THROTTLE_RESULTS = (6.0, 10.0)  # pause between result pages
THROTTLE_TITLES  = (3.0, 6.0)   # pause between title pages

# ---------- Model ----------
@dataclass
class MovieRow:
    title: str
    storyline: str
    url: str
    imdb_id: str

# ---------- Driver ----------
def make_driver(headless: bool = False):
    """Non-headless + persistent profile helps bypass 403; uc handles stealth."""
    w = random.randint(1200, 1600)
    h = random.randint(800, 1100)
    opts = uc.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument(f"--user-data-dir={PROFILE_DIR}")  # keep cookies/consent
    opts.add_argument(f"--window-size={w},{h}")
    opts.add_argument("--lang=en-US")
    opts.add_argument(f"--user-agent={random.choice(UA_LIST)}")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    driver = uc.Chrome(options=opts)
    # Best-effort mask
    try:
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
        })
    except Exception:
        pass
    return driver

# ---------- Helpers ----------
def search_page_url(start: int) -> str:
    return f"{BASE_URL}&start={start}"  # 1, 51, 101, ...

def parse_imdb_id_from_url(url: str) -> str:
    try:
        return url.split("/title/")[1].split("/")[0]
    except Exception:
        return ""

def human_pause(lo, hi):
    time.sleep(random.uniform(lo, hi))

def accept_consent_if_present(driver):
    try:
        for sel in [
            '[data-testid="consent-banner-accept"]',
            '#onetrust-accept-btn-handler',
            'button[aria-label*="Accept"]',
        ]:
            btns = driver.find_elements(By.CSS_SELECTOR, sel)
            if btns:
                btns[0].click()
                time.sleep(0.5)
                break
    except Exception:
        pass

def is_blocked_403(driver) -> bool:
    try:
        title = (driver.title or "").lower()
        if "403" in title or "forbidden" in title:
            return True
        body = driver.find_element(By.TAG_NAME, "body").text.lower()
        return ("403" in body) and ("forbidden" in body)
    except Exception:
        return False

def load_with_recovery(url: str, driver, max_restarts=2) -> uc.Chrome | None:
    """Open URL; if 403, backoff and restart driver up to max_restarts."""
    for attempt in range(1, max_restarts + 1):
        driver.get(url)
        accept_consent_if_present(driver)
        human_pause(1.5, 3.0)
        if not is_blocked_403(driver):
            return driver
        print(f"  ! 403 Forbidden on attempt {attempt}. Backing off...")
        time.sleep(60 * attempt)   # backoff: 60s, 120s, ...
        try:
            driver.quit()
        except Exception:
            pass
        driver = make_driver(headless=False)
    return None

def extract_storyline_from_html(html: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    # 1) JSON-LD
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            import json as _json
            if not script.string:
                continue
            j = _json.loads(script.string)
            def pick_desc(obj):
                if isinstance(obj, dict):
                    d = obj.get("description")
                    if d and len(d.strip()) > 20:
                        return d.strip()
            if isinstance(j, list):
                for obj in j:
                    d = pick_desc(obj)
                    if d: return d
            else:
                d = pick_desc(j)
                if d: return d
        except Exception:
            pass
    # 2) OpenGraph
    og = soup.find("meta", {"property": "og:description"})
    if og and og.get("content"):
        d = og["content"].strip()
        if d and len(d) > 20:
            return d
    # 3) Plot testids
    plot = soup.select_one('[data-testid="plot-xl"], [data-testid="plot-l"], [data-testid="plot"]')
    if plot:
        txt = plot.get_text(strip=True)
        if txt and len(txt) > 20:
            return txt
    # 4) Storyline heading
    for h in soup.find_all(["h2","h3"]):
        if "storyline" in h.get_text(" ", strip=True).lower():
            nxt = h.find_next()
            if nxt:
                txt = nxt.get_text(" ", strip=True)
                if txt and len(txt) > 20:
                    return txt
    return None

def append_rows(rows: list[MovieRow]):
    mode = "a" if OUT_CSV.exists() else "w"
    with OUT_CSV.open(mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["title","storyline","url","imdb_id"])
        if mode == "w":
            w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

# ---------- Core ----------
def scrape_result_links(driver, start: int) -> list[str]:
    url = search_page_url(start)
    driver = load_with_recovery(url, driver)
    if driver is None:
        print("  ! Could not bypass 403 on results page.")
        return []

    # Wait for either layout
    try:
        WebDriverWait(driver, 15).until(
            EC.any_of(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "h3.lister-item-header a")),
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'a[href*="/title/tt"].ipc-metadata-list-summary-item__t')),
            )
        )
    except TimeoutException:
        print("  ! Results not visible. Title:", driver.title)
        return []

    links = []
    # Old lister
    for a in driver.find_elements(By.CSS_SELECTOR, "h3.lister-item-header a"):
        href = a.get_attribute("href")
        if href and "/title/" in href:
            links.append(href.split("?")[0])
    # New ipc
    for a in driver.find_elements(By.CSS_SELECTOR, 'a[href*="/title/tt"].ipc-metadata-list-summary-item__t'):
        href = a.get_attribute("href")
        if href and "/title/" in href:
            links.append(href.split("?")[0])

    # Dedup preserve order
    seen, uniq = set(), []
    for l in links:
        if l not in seen:
            uniq.append(l); seen.add(l)
    return uniq

def scrape_all_2024(headless=False):
    driver = make_driver(headless=headless)  # start non-headless
    try:
        # Resume support
        seen_ids = set()
        if OUT_CSV.exists():
            try:
                df = pd.read_csv(OUT_CSV)
                seen_ids = set(df.get("imdb_id", pd.Series(dtype=str)).dropna().astype(str).tolist())
                print(f"↺ Resume: {len(seen_ids)} rows already in {OUT_CSV}")
            except Exception:
                pass

        start = 1
        total_new = 0

        while True:
            print(f"\n[Search] start={start}")
            links = scrape_result_links(driver, start)
            if not links:
                print("No more results (or failed to load). Stopping.")
                break

            buffer: list[MovieRow] = []
            for idx, url in enumerate(links, 1):
                imdb_id = parse_imdb_id_from_url(url)
                if imdb_id in seen_ids:
                    print(f"  • [{idx:03d}] skip (seen) {imdb_id}")
                    human_pause(*THROTTLE_TITLES)
                    continue

                # Load title page with recovery
                d2 = load_with_recovery(url, driver)
                if d2 is None:
                    print(f"  x [{idx:03d}] 403 persists on title page, skipping.")
                    human_pause(*THROTTLE_TITLES)
                    continue
                driver = d2  # keep the possibly restarted driver

                try:
                    WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, "title")))
                except TimeoutException:
                    print(f"  ! [{idx:03d}] title didn't load, skip.")
                    human_pause(*THROTTLE_TITLES)
                    continue

                html = driver.page_source

                # Title
                title = ""
                try:
                    soup = BeautifulSoup(html, "lxml")
                    jd = soup.find("script", {"type": "application/ld+json"})
                    if jd and jd.string:
                        import json as _json
                        d = _json.loads(jd.string)
                        if isinstance(d, dict):
                            title = d.get("name") or ""
                        elif isinstance(d, list):
                            for obj in d:
                                if isinstance(obj, dict) and obj.get("@type") in {"Movie","TVSeries"}:
                                    title = obj.get("name") or ""
                                    if title: break
                    if not title:
                        ttag = soup.find("title")
                        if ttag:
                            title = ttag.get_text(strip=True).replace("- IMDb","").strip()
                except Exception:
                    pass

                storyline = extract_storyline_from_html(html) or ""
                if title and storyline:
                    buffer.append(MovieRow(title=title, storyline=storyline, url=url, imdb_id=imdb_id))
                    seen_ids.add(imdb_id)
                    total_new += 1
                    print(f"  ✓ [{idx:03d}] {title} ({imdb_id})")
                else:
                    print(f"  · [{idx:03d}] skipped (missing title/storyline): {imdb_id}")

                human_pause(*THROTTLE_TITLES)

                # Flush every 40 items
                if len(buffer) >= 40:
                    append_rows(buffer)
                    print(f"    ↳ saved {len(buffer)} rows (total new={total_new})")
                    buffer.clear()

            if buffer:
                append_rows(buffer)
                print(f"    ↳ saved {len(buffer)} rows (page end; total new={total_new})")

            start += 50                   # next page (count=50)
            human_pause(*THROTTLE_RESULTS)

        # Final dedup
        if OUT_CSV.exists():
            df = pd.read_csv(OUT_CSV)
            before = len(df)
            df = df.drop_duplicates(subset=["imdb_id"]).reset_index(drop=True)
            df.to_csv(OUT_CSV, index=False)
            print(f"\nDe-duplicated: {before} -> {len(df)} rows in {OUT_CSV}")

        print(f"\nDone. New rows collected: {total_new}")

    finally:
        try:
            driver.quit()
        except Exception:
            pass

if __name__ == "__main__":
    scrape_all_2024(headless=False)  # start non-headless first time
