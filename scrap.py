#!/usr/bin/env python3
import csv, random, time, calendar
from pathlib import Path
from dataclasses import dataclass, asdict
from urllib.parse import urlencode

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, WebDriverException,
    ElementClickInterceptedException, StaleElementReferenceException
)
from webdriver_manager.chrome import ChromeDriverManager

# ---------------- CONFIG ----------------
YEAR = 2024
TITLE_TYPE = "feature"      # IMDb: "Movie" == feature
SORT = "moviemeter,asc"
OUT_CSV = Path("imdb_2024_list_all.csv")

PAGE_TIMEOUT = 15
THROTTLE = (0.6, 1.2)
CLICK_PAUSE = (0.4, 0.8)
MAX_CLICKS_PER_SLICE = 10000  # essentially "until no button"

BASE = "https://www.imdb.com/search/title/"

# ---------- Selectors (new & old layouts) ----------
SEL_NEW_ITEM   = "li.ipc-metadata-list-summary-item"
SEL_NEW_LINK   = 'a.ipc-metadata-list-summary-item__t[href*="/title/tt"], a[href*="/title/tt"]'
SEL_NEW_TITLE  = "h3.ipc-title__text"
SEL_NEW_RATING = "span.ipc-rating-star--rating"
SEL_NEW_VOTES  = "span.ipc-rating-star--voteCount"
SEL_NEW_TIME   = "div.dli-title-metadata span"
SEL_50_MORE    = [
    "button.ipc-btn--load-more",
    "button.ipc-see-more__button",
    # “50 more” text variants (case-insensitive)
    "//span[contains(translate(., 'MORE', 'more'),'50 more')]/ancestor::button[1]",
    "//button[contains(., '50 more')]",
]

SEL_OLD_ITEM   = "div.lister-item.mode-advanced"
SEL_OLD_LINK   = "h3.lister-item-header a"
SEL_OLD_RATING = "div.ratings-bar strong, span.ipl-rating-star__rating"
SEL_OLD_VOTES  = "p.sort-num_votes-visible span[name='nv']"
SEL_OLD_TIME   = "p.text-muted span.runtime"

# ---------------- Driver -----------------
def make_driver(headless=False):
    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1400,1000")
    opts.add_argument("--lang=en-US")
    # soften automation signatures a bit
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    try:
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        })
    except Exception:
        pass
    return driver

def throttle():
    time.sleep(random.uniform(*THROTTLE))

def short_pause():
    time.sleep(random.uniform(*CLICK_PAUSE))

def accept_consent_if_present(driver):
    try:
        for sel in (
            '[data-testid="consent-banner-accept"]',
            '#onetrust-accept-btn-handler',
            'button[aria-label*="Accept"]',
            'button[aria-label*="accept"]',
        ):
            els = driver.find_elements(By.CSS_SELECTOR, sel)
            if els:
                els[0].click()
                time.sleep(0.3)
                break
    except Exception:
        pass

def wait_for_any_layout(driver):
    WebDriverWait(driver, PAGE_TIMEOUT).until(
        EC.any_of(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, SEL_NEW_ITEM)),
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, SEL_OLD_ITEM)),
        )
    )

# --------------- Helpers ----------------
def month_slices(year: int):
    for m in range(1, 13):
        last = calendar.monthrange(year, m)[1]
        yield f"{year}-{m:02d}-01", f"{year}-{m:02d}-{last:02d}"

def build_month_url(date_from: str, date_to: str, start: int) -> str:
    # count=50 is important to expose the “50 more” button consistently
    qs = {
        "title_type": TITLE_TYPE,
        "release_date": f"{date_from},{date_to}",
        "view": "advanced",
        "count": "50",
        "sort": SORT,
        "start": str(start),
    }
    return BASE + "?" + urlencode(qs)

def normalize_title(raw: str) -> str:
    raw = raw.strip()
    if ". " in raw and raw.split(". ", 1)[0].isdigit():
        return raw.split(". ", 1)[1]
    return raw

def parse_id_from_url(url: str) -> str:
    try:
        return url.split("/title/")[1].split("/")[0]
    except Exception:
        return ""

def get_all_cards(driver):
    cards = driver.find_elements(By.CSS_SELECTOR, SEL_NEW_ITEM)
    layout = "new"
    if not cards:
        cards = driver.find_elements(By.CSS_SELECTOR, SEL_OLD_ITEM)
        layout = "old"
    return cards, layout

def find_50_more(driver):
    # CSS first
    for css in SEL_50_MORE[:2]:
        btns = driver.find_elements(By.CSS_SELECTOR, css)
        if btns:
            return ("css", css, btns[0])
    # XPATH fallbacks
    for xp in SEL_50_MORE[2:]:
        try:
            btns = driver.find_elements(By.XPATH, xp)
            if btns:
                return ("xpath", xp, btns[0])
        except Exception:
            pass
    return None

def pick_blurb_old(li) -> str:
    try:
        ps = li.find_elements(By.CSS_SELECTOR, ".lister-item-content p")
        for p in ps:
            txt = p.text.strip()
            low = txt.lower()
            if len(txt) > 20 and not any(k in low for k in ("director", "star", "metascore", "votes")):
                return txt
    except Exception:
        pass
    return ""

def pick_blurb_new(li) -> str:
    try:
        t = li.find_element(By.CSS_SELECTOR, "div.ipc-html-content-inner-div").get_attribute("innerText").strip()
        if len(t) > 20:
            return t
    except Exception:
        pass
    return ""

def parse_cards_into_rows(driver, seen_global: set, seen_slice: set) -> list[dict]:
    rows = []
    cards, layout = get_all_cards(driver)

    for li in cards:
        try:
            if layout == "new":
                a = li.find_element(By.CSS_SELECTOR, SEL_NEW_LINK)
                url = a.get_attribute("href").split("?")[0]
                try:
                    title = normalize_title(li.find_element(By.CSS_SELECTOR, SEL_NEW_TITLE).text)
                except Exception:
                    title = a.text.strip()
                # rating / votes / time
                try:
                    rating = li.find_element(By.CSS_SELECTOR, SEL_NEW_RATING).text.strip()
                except Exception:
                    rating = ""
                try:
                    votes = li.find_element(By.CSS_SELECTOR, SEL_NEW_VOTES).text.strip("()").replace(",", "")
                except Exception:
                    votes = ""
                try:
                    spans = li.find_elements(By.CSS_SELECTOR, SEL_NEW_TIME)
                    duration = spans[1].get_attribute("innerText").strip() if len(spans) >= 2 else ""
                except Exception:
                    duration = ""
                blurb = pick_blurb_new(li)
            else:
                a = li.find_element(By.CSS_SELECTOR, SEL_OLD_LINK)
                url = a.get_attribute("href").split("?")[0]
                title = normalize_title(a.text)
                try:
                    rating = li.find_element(By.CSS_SELECTOR, SEL_OLD_RATING).text.strip()
                except Exception:
                    rating = ""
                votes = ""
                try:
                    nv = li.find_elements(By.CSS_SELECTOR, SEL_OLD_VOTES)
                    if nv:
                        votes = (nv[0].get_attribute("data-value") or nv[0].text).replace(",", "")
                except Exception:
                    pass
                try:
                    duration = li.find_element(By.CSS_SELECTOR, SEL_OLD_TIME).text.strip()
                except Exception:
                    duration = ""
                blurb = pick_blurb_old(li)
        except Exception:
            continue

        imdb_id = parse_id_from_url(url)
        if not imdb_id or imdb_id in seen_global or imdb_id in seen_slice:
            continue

        rows.append({
            "Movie Name": title,
            "IMDb ID": imdb_id,
            "URL": url,
            "Rating": rating,
            "Voting Counts": votes,
            "Duration": duration,
            "Storyline (list blurb)": blurb,
        })
    return rows

def append_rows(rows: list[dict]):
    mode = "a" if OUT_CSV.exists() else "w"
    with OUT_CSV.open(mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "Movie Name","IMDb ID","URL","Rating","Voting Counts","Duration","Storyline (list blurb)"
        ])
        if mode == "w":
            w.writeheader()
        w.writerows(rows)

# ------------- Core per-slice -------------
def scrape_month_with_load_more(driver, date_from: str, date_to: str, seen_global: set) -> int:
    """Loads monthly page, then keeps clicking '50 more' until the button disappears.
       Saves after each click so virtualization/crashes don’t lose progress.
       If tab crashes, resumes from the correct start offset.
    """
    saved_in_slice = 0
    seen_slice = set()

    # resume loop: if we crash, we reload with start = saved_in_slice + 1
    while True:
        start_offset = saved_in_slice + 1
        url = build_month_url(date_from, date_to, start_offset)

        # nav with crash-guard
        try:
            driver.get(url)
        except WebDriverException:
            try:
                driver.quit()
            except Exception:
                pass
            driver = make_driver(headless=False)
            driver.get(url)

        accept_consent_if_present(driver)
        try:
            wait_for_any_layout(driver)
        except TimeoutException:
            # month likely empty; bail
            return saved_in_slice

        # initial parse & save (the first 50 at this offset)
        rows = parse_cards_into_rows(driver, seen_global, seen_slice)
        if rows:
            append_rows(rows)
            for r in rows:
                seen_global.add(r["IMDb ID"]); seen_slice.add(r["IMDb ID"])
            saved_in_slice += len(rows)
            print(f"📦 [{date_from}..{date_to}] Saved {len(rows)} rows (slice total: {saved_in_slice})")
        else:
            print(f"ℹ️ [{date_from}..{date_to}] No new rows at start={start_offset}")

        clicks = 0
        while clicks < MAX_CLICKS_PER_SLICE:
            # find button
            btn_triplet = find_50_more(driver)
            if not btn_triplet:
                print("✅ Button gone for this month — fully expanded at current offset.")
                break

            # scroll toward bottom to ensure virtualization shows the “end”
            try:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 200);")
            except Exception:
                pass
            short_pause()

            # click with fallbacks
            _, _, btn = btn_triplet
            clicked = False
            try:
                btn.click()
                clicked = True
            except (ElementClickInterceptedException, StaleElementReferenceException):
                try:
                    driver.execute_script("arguments[0].click();", btn)
                    clicked = True
                except Exception:
                    clicked = False
            except Exception:
                try:
                    driver.execute_script("arguments[0].click();", btn)
                    clicked = True
                except Exception:
                    clicked = False

            if not clicked:
                print("⚠️ Click failed; stopping this offset chunk.")
                break

            # let items load a moment
            short_pause()

            # parse & save *whatever is in DOM now* (handles virtualization)
            before = saved_in_slice
            rows = parse_cards_into_rows(driver, seen_global, seen_slice)
            if rows:
                append_rows(rows)
                for r in rows:
                    seen_global.add(r["IMDb ID"]); seen_slice.add(r["IMDb ID"])
                saved_in_slice += len(rows)
                print(f"➡️ Clicked '50 more' ({clicks+1}). Saved {len(rows)} new rows (slice total: {saved_in_slice})")
            else:
                print("ℹ️ Click registered but no new unique rows this pass (likely virtualization).")

            clicks += 1
            throttle()

        # If button disappeared, the current offset is fully drained.
        # We now try to continue from the *next* offset (saved_in_slice + 1).
        # If that next page yields no new rows immediately, we’re done with the month.
        # Otherwise the outer while True will repeat, adding more until exhausted.
        # The outer loop breaks when first page after an offset saves 0 new rows.
        # That means “no more results at all for this month”.
        # Outer while will re-check; if empty at fresh offset, break:
        rows_test_url = build_month_url(date_from, date_to, saved_in_slice + 1)
        try:
            driver.get(rows_test_url)
            accept_consent_if_present(driver)
            wait_for_any_layout(driver)
            probe = parse_cards_into_rows(driver, seen_global, seen_slice)
            if not probe:
                break
            else:
                # We found more beyond this offset; loop will repeat with new offset.
                continue
        except Exception:
            break

    return saved_in_slice

# ----------------- Runner -----------------
def scrape_all_from_listing():
    # resume from CSV across slices
    seen_global = set()
    if OUT_CSV.exists():
        try:
            df = pd.read_csv(OUT_CSV)
            seen_global = set(df.get("IMDb ID", pd.Series(dtype=str)).dropna().astype(str))
            print(f"↺ Resume: {len(seen_global)} rows already in {OUT_CSV}")
        except Exception:
            pass

    total_saved = 0
    driver = make_driver(headless=False)
    try:
        for date_from, date_to in month_slices(YEAR):
            print(f"\n==> Slice {date_from} .. {date_to}")
            try:
                added = scrape_month_with_load_more(driver, date_from, date_to, seen_global)
            except WebDriverException:
                print("💥 Tab crashed. Recovering and resuming this slice…")
                try:
                    driver.quit()
                except Exception:
                    pass
                driver = make_driver(headless=False)
                added = scrape_month_with_load_more(driver, date_from, date_to, seen_global)

            print(f"   ↳ Finished slice {date_from}..{date_to} (+{added})")
            total_saved += added
            time.sleep(1.0)
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    # final dedupe
    if OUT_CSV.exists():
        df = pd.read_csv(OUT_CSV)
        before = len(df)
        df = df.drop_duplicates(subset=["IMDb ID"]).reset_index(drop=True)
        df.to_csv(OUT_CSV, index=False)
        print(f"\n🔁 De-duplicated: {before} -> {len(df)} rows in {OUT_CSV}")

    print(f"Done. Collected {total_saved} NEW rows. CSV: {OUT_CSV}")

if __name__ == "__main__":
    scrape_all_from_listing()
