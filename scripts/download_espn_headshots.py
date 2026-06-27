import json
import re
import time
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz


ROOT = Path(".")
PROSPECTS_JSON = ROOT / "frontend/public/data/prospects.json"
OUT_DIR = ROOT / "frontend/public/headshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/html,*/*",
}


def slugify(name):
    s = str(name).lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def download(url, path):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return False
        if not r.content or len(r.content) < 1000:
            return False

        content_type = r.headers.get("content-type", "").lower()
        if "image" not in content_type:
            return False

        path.write_bytes(r.content)
        return True
    except Exception:
        return False


def search_espn_site(player_name):
    """
    Search ESPN's site search endpoint.
    This endpoint can change, so the script also has fallback methods.
    """
    urls = [
        f"https://site.web.api.espn.com/apis/search/v2?query={quote_plus(player_name)}&limit=10",
        f"https://site.web.api.espn.com/apis/search/v2?region=us&lang=en&query={quote_plus(player_name)}&limit=10",
    ]

    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code != 200:
                continue

            data = r.json()
            text = json.dumps(data)

            ids = sorted(set(re.findall(r'athlete[s]?/(\d+)', text)))
            page_ids = sorted(set(re.findall(r'/id/(\d+)', text)))

            all_ids = list(dict.fromkeys(ids + page_ids))
            if all_ids:
                return all_ids
        except Exception:
            continue

    return []


def get_athlete_json(athlete_id):
    urls = [
        f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/athletes/{athlete_id}?lang=en&region=us",
        f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/athletes/{athlete_id}?lang=en&region=us",
    ]

    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass

    return None


def extract_headshot_from_athlete_json(data):
    if not isinstance(data, dict):
        return None

    # Common ESPN athlete JSON field
    if isinstance(data.get("headshot"), dict):
        href = data["headshot"].get("href")
        if href:
            return href

    # Sometimes image/headshot URLs appear nested
    text = json.dumps(data)
    candidates = re.findall(r'https?://[^"\']+\.(?:png|jpg|jpeg)(?:\?[^"\']*)?', text)

    # Prefer ESPN CDN images
    candidates = [c.replace("\\/", "/") for c in candidates]
    espn_candidates = [c for c in candidates if "espncdn" in c or "a.espncdn.com" in c]

    return espn_candidates[0] if espn_candidates else (candidates[0] if candidates else None)


def espn_static_headshot_candidates(athlete_id):
    """
    ESPN often uses static athlete image patterns based on athlete ID.
    Try common sizes/formats.
    """
    return [
        f"https://a.espncdn.com/i/headshots/mens-college-basketball/players/full/{athlete_id}.png",
        f"https://a.espncdn.com/i/headshots/mens-college-basketball/players/full/{athlete_id}.jpg",
        f"https://a.espncdn.com/i/headshots/nba/players/full/{athlete_id}.png",
        f"https://a.espncdn.com/i/headshots/nba/players/full/{athlete_id}.jpg",
        f"https://a.espncdn.com/i/headshots/recruiting/ncb/players/full/{athlete_id}.png",
        f"https://a.espncdn.com/i/headshots/recruiting/ncb/players/full/{athlete_id}.jpg",
    ]


def search_web_espn_page(player_name):
    """
    Fallback: use ESPN search HTML result pages and pull likely /id/ URLs.
    """
    url = f"https://www.espn.com/search/_/q/{quote_plus(player_name)}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return []

        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(" ")
        html = str(soup)

        ids = sorted(set(re.findall(r'/id/(\d+)', html)))
        return ids
    except Exception:
        return []


def best_id_for_player(player_name):
    ids = search_espn_site(player_name)

    if not ids:
        ids = search_web_espn_page(player_name)

    best = None
    best_score = -1

    for athlete_id in ids:
        data = get_athlete_json(athlete_id)
        if not data:
            continue

        display_name = data.get("displayName") or data.get("fullName") or data.get("name") or ""
        score = fuzz.token_sort_ratio(player_name.lower(), str(display_name).lower())

        if score > best_score:
            best_score = score
            best = {
                "athlete_id": athlete_id,
                "display_name": display_name,
                "score": score,
                "data": data,
            }

    if best and best["score"] >= 70:
        return best

    # If JSON lookup failed but search found ids, still try static URLs.
    if ids:
        return {
            "athlete_id": ids[0],
            "display_name": "",
            "score": None,
            "data": None,
        }

    return None


def main():
    if not PROSPECTS_JSON.exists():
        raise SystemExit(f"Missing {PROSPECTS_JSON}. Run scripts/export_frontend_data.py first.")

    prospects = pd.read_json(PROSPECTS_JSON)
    results = []

    for i, row in prospects.iterrows():
        name = row["player_name"]
        slug = row.get("slug") or slugify(name)

        out_path = OUT_DIR / f"{slug}.jpg"
        if out_path.exists() and out_path.stat().st_size > 1000:
            print(f"[SKIP] {name}: already exists")
            results.append({"player_name": name, "status": "already_exists", "file": str(out_path)})
            continue

        print(f"[{i+1}/{len(prospects)}] Searching {name}...")

        match = best_id_for_player(name)
        if not match:
            print(f"  NO ESPN ID FOUND")
            results.append({"player_name": name, "status": "no_id"})
            time.sleep(0.5)
            continue

        athlete_id = match["athlete_id"]
        headshot_url = None

        if match.get("data"):
            headshot_url = extract_headshot_from_athlete_json(match["data"])

        urls_to_try = []
        if headshot_url:
            urls_to_try.append(headshot_url)

        urls_to_try.extend(espn_static_headshot_candidates(athlete_id))

        ok = False
        for url in urls_to_try:
            if download(url, out_path):
                print(f"  SAVED {out_path.name} from {url}")
                results.append(
                    {
                        "player_name": name,
                        "status": "saved",
                        "espn_id": athlete_id,
                        "espn_name": match.get("display_name"),
                        "match_score": match.get("score"),
                        "url": url,
                        "file": str(out_path),
                    }
                )
                ok = True
                break

        if not ok:
            print(f"  NO HEADSHOT FOUND for ESPN ID {athlete_id}")
            results.append(
                {
                    "player_name": name,
                    "status": "no_headshot",
                    "espn_id": athlete_id,
                    "espn_name": match.get("display_name"),
                    "match_score": match.get("score"),
                }
            )

        time.sleep(0.5)

    pd.DataFrame(results).to_csv(ROOT / "reports/espn_headshot_download_report.csv", index=False)
    print("\nSaved reports/espn_headshot_download_report.csv")
    print(f"Images saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
