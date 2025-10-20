import os
import time
import random
import re
import json
from urllib.parse import urljoin
from bs4 import BeautifulSoup, Comment
from playwright.sync_api import sync_playwright

# -----------------------------
# Config
# -----------------------------
BASE_DIR = "Premier_League_2024_2025_2025_2026"
PLAYER_TABS = ["summary", "passing", "passing_types", "defense", "possession", "misc", "keeper", "keeper_adv"]

# -----------------------------
# Helper functions
# -----------------------------

def table_to_dicts(table):
    """Convert a BeautifulSoup table to a list of dictionaries"""
    thead = table.find("thead")
    if thead:
        headers = [th.get_text(strip=True) for th in thead.find_all("tr")[-1].find_all("th")]
    else:
        first_row = table.find("tbody").find("tr")
        headers = [th.get_text(strip=True) for th in first_row.find_all(["th","td"])] if first_row else []

    rows = []
    tbody = table.find("tbody")
    if not tbody:
        return rows

    for tr in tbody.find_all("tr"):
        if tr.get("class") and "thead" in tr.get("class"):
            continue
        cells = [td.get_text(strip=True) for td in tr.find_all(["th","td"])]
        if len(cells) < len(headers):
            cells += [""]*(len(headers)-len(cells))
        elif len(cells) > len(headers):
            cells = cells[:len(headers)]
        rows.append(dict(zip(headers, cells)))
    return rows

def choose_matchlog_table(tables):
    """Pick the most likely matchlog table from a list"""
    scored = []
    for t in tables:
        tid = (t.get("id") or "").lower()
        caption = (t.find("caption").get_text(" ", strip=True) if t.find("caption") else "").lower()
        thead = t.find("thead")
        headers = []
        if thead:
            last = thead.find_all("tr")[-1]
            headers = [th.get_text(strip=True).lower() for th in last.find_all("th")]
        header_text = " ".join(headers)

        score = 0
        if "matchlog" in tid: score += 6
        if "match log" in caption or "matchlogs" in caption: score += 6
        if any(x in header_text for x in ("date","min","goals","result","comp","opponent","xg")):
            score += 4
        if "premier" in caption or "premier" in header_text:
            score += 2
        scored.append((score, t))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored[0][0] > 0 else scored[0][1]

def extract_table_rows(table):
    """Convert table to list of row dicts"""
    thead = table.find("thead")
    if not thead:
        return []
    headers = [th.get_text(strip=True) for th in thead.find_all("tr")[-1].find_all("th")]

    tbody = table.find("tbody")
    if not tbody:
        return []

    rows = []
    for tr in tbody.find_all("tr"):
        if tr.get("class") and "thead" in tr.get("class"):
            continue
        cells = [td.get_text(strip=True) for td in tr.find_all(["th","td"])]
        if len(cells) < len(headers):
            cells += [""]*(len(headers)-len(cells))
        elif len(cells) > len(headers):
            cells = cells[:len(headers)]
        rows.append(dict(zip(headers, cells)))
    return rows

def find_matchlogs_link(soup):
    """Locate a matchlogs link on player page, including in commented blocks"""
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "matchlogs" in href:
            return urljoin("https://fbref.com", href)
    for c in soup.find_all(string=lambda text: isinstance(text, Comment)):
        csoup = BeautifulSoup(c, "html.parser")
        for a in csoup.find_all("a", href=True):
            href = a["href"]
            if "matchlogs" in href:
                return urljoin("https://fbref.com", href)
    return None

def extract_height_weight(soup):
    height = ""
    weight = ""
    footed = ""

    # Look in #meta div first
    meta_div = soup.find("div", id="meta")
    if not meta_div:
        # fallback: search inside HTML comments
        for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
            csoup = BeautifulSoup(c, "html.parser")
            meta_div = csoup.find("div", id="meta")
            if meta_div:
                break

    if meta_div:
        for p in meta_div.find_all("p"):
            strongs = p.find_all("strong")
            for s in strongs:
                text = s.get_text(strip=True)
                content = s.next_sibling
                if content:
                    content = content.strip()
                if "Footed:" in text and content:
                    footed = content.split()[0]  # not used currently   
            spans = p.find_all("span")
            if len(spans) >= 2:
                height = spans[0].get_text(strip=True).replace("cm","").strip()
                weight = spans[1].get_text(strip=True).replace("kg","").strip()
                break  # stop after first match

    return height, weight, footed



# -----------------------------
# Scraper functions
# -----------------------------

def scrape_player_tabs(player_name, player_profile_url, team_dir, page):
    safe_player_name = player_name.replace("/", "-")
    player_dir = os.path.join(team_dir, safe_player_name)
    os.makedirs(player_dir, exist_ok=True)

    page.goto(player_profile_url)
    time.sleep(random.uniform(2,4))
    soup = BeautifulSoup(page.content(), "html.parser")

    height,weight,footed = "","",""
    height, weight, footed = extract_height_weight(soup)

    info_file = os.path.join(player_dir, "info.json")
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump({"name": player_name, "height": height, "weight": weight, "Footed" : footed}, f, indent=2, ensure_ascii=False)
    print(f"       ✅ Saved info.json (Height: {height}cm, Weight: {weight}kg)")

    matchlog_link = find_matchlogs_link(soup)

    for tab in PLAYER_TABS:
        try:
            if matchlog_link and "/matchlogs/" in matchlog_link:
                tab_url = re.sub(r"/matchlogs/\d{4}-\d{4}/", "/matchlogs/2024-2025/", matchlog_link)
                tab_url = tab_url.replace("/matchlogs/2024-2025/", f"/matchlogs/2024-2025/{tab}/")
            else:
                print(f"       ⚠️ No dedicated link for {tab}, skipping")
                continue

            page.goto(tab_url)
            time.sleep(random.uniform(2,4))
            tab_soup = BeautifulSoup(page.content(), "html.parser")

            all_tables = tab_soup.find_all("table")
            for c in tab_soup.find_all(string=lambda text: isinstance(text, Comment)):
                all_tables.extend(BeautifulSoup(c, "html.parser").find_all("table"))

            chosen = choose_matchlog_table(all_tables)
            if chosen:
                rows = extract_table_rows(chosen)
            else:
                rows = []

            # Save JSON
            filename = os.path.join(player_dir, f"{tab}_all.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)    

            print(f"       ✅ Saved {tab}_all.json ({len(rows)} rows)")

            # Filter to Premier League if comp column exists
            if rows:
                comp_field = None
                for k in rows[0].keys():
                    if "comp" in k.lower() or "competition" in k.lower():
                        comp_field = k
                        break
                if comp_field:
                    rows = [r for r in rows if "premier" in (r.get(comp_field) or "").lower()]

            # Save JSON
            filename = os.path.join(player_dir, f"{tab}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)

            print(f"       ✅ Saved {tab}.json ({len(rows)} rows)")

            time.sleep(random.uniform(1.5,3))

        except Exception as e:
            print(f"       ❌ Error scraping tab {tab} for {player_name}: {e}")

def scrape_team(team_name, team_url, page):
    safe_team = team_name.replace("/", "-")
    team_dir = os.path.join(BASE_DIR, safe_team)
    os.makedirs(team_dir, exist_ok=True)

    page.goto(team_url)
    time.sleep(random.uniform(3,5))
    soup = BeautifulSoup(page.content(), "html.parser")

    table = soup.find("table", id="stats_standard_9")
    if not table:
        for c in soup.find_all(string=lambda text: isinstance(text, Comment)):
            csoup = BeautifulSoup(c, "html.parser")
            table = csoup.find("table", id="stats_standard_9")
            if table:
                break
    if not table:
        print(f"❌ No player table found for {team_name}")
        return

    tbody = table.find("tbody")
    for tr in tbody.find_all("tr"):
        if tr.get("class") and "thead" in tr.get("class"):
            continue
        player_th = tr.find("th", {"data-stat": "player"})
        if not player_th:
            continue
        a = player_th.find("a", href=True)
        if not a:
            continue
        player_name = player_th.get_text(strip=True)
        player_profile_url = urljoin("https://fbref.com", a["href"])
        print(f"  -> Player: {player_name}")
        scrape_player_tabs(player_name, player_profile_url, team_dir, page)
        time.sleep(random.uniform(1.5,3))

def scrape_premier_league():
    os.makedirs(BASE_DIR, exist_ok=True)
    league_url = "https://fbref.com/en/comps/9/2025-2026/2025-2026-Premier-League-Stats"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(league_url)
        time.sleep(random.uniform(3,5))
        soup = BeautifulSoup(page.content(), "html.parser")

        # Get team links (including inside comments)
        team_links = []
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for c in comments:
            csoup = BeautifulSoup(c, "html.parser")
            team_links.extend(csoup.select("table a[href*='/squads/']"))
        if not team_links:
            team_links = soup.select("table a[href*='/squads/']")

        teams = {}
        for a in team_links:
            href = a["href"]
            full_url = urljoin("https://fbref.com", href)
            team_name = a.get_text(strip=True)
            teams[team_name] = full_url

        print(f"🏆 Found {len(teams)} teams")

        for team_name, team_url in teams.items():
            team_name_clean = team_name.replace("\xa0", " ").lower()
            if team_name_clean not in ["chelsea", "burnley","brentford","arsenal","bournemouth","manchester city","fulham","brighton","aston villa","tottenham","crystal palace","sunderland","newcastle utd","west ham","everton","liverpool"]:
                print(f"\n=== Team: {team_name} ===")
                scrape_team(team_name, team_url, page)
                time.sleep(random.uniform(5,10))

        browser.close()

if __name__ == "__main__":
    scrape_premier_league()
