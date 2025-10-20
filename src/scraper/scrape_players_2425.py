import os
import time
import random
import re
import json
from urllib.parse import urljoin
from bs4 import BeautifulSoup, Comment
from playwright.sync_api import sync_playwright

# Helper: find a matchlogs link on a player page (including in commented blocks)
def find_matchlogs_link(soup):
    # 1) look for hrefs containing "matchlogs"
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "matchlogs" in href:
            return urljoin("https://fbref.com", href)
        if re.search(r"match\s*log", a.get_text(), re.I):
            return urljoin("https://fbref.com", href)
    # 2) search inside HTML comments
    for c in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment_soup = BeautifulSoup(c, "html.parser")
        for a in comment_soup.find_all("a", href=True):
            href = a["href"]
            if "matchlogs" in href or re.search(r"match\s*log", a.get_text(), re.I):
                return urljoin("https://fbref.com", href)
    return None

# Helper: pick best candidate table that looks like a match log
def choose_matchlog_table(tables):
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
    # Return highest scoring table if it has score > 0, else return first table
    return scored[0][1] if scored[0][0] > 0 else scored[0][1]

# Extract rows from a chosen table and optionally filter to Premier League 2024-25
def extract_table_rows(table):
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
        cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
        if not cells or len(cells) == 0:
            continue
        # If number of cells mismatches headers, try to pad/truncate safely
        if len(cells) < len(headers):
            cells += [""] * (len(headers) - len(cells))
        elif len(cells) > len(headers):
            cells = cells[:len(headers)]
        rows.append(dict(zip(headers, cells)))
    return rows

# Main scraping flow
def scrape_premier_league_players_matchlogs():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)   # set False if blocked
        context = browser.new_context()
        page = context.new_page()

        # Premier League 2024-25 overview (season page)
        league_url = "https://fbref.com/en/comps/9/2024-2025/2024-2025-Premier-League-Stats"
        page.goto(league_url)
        time.sleep(5 + random.uniform(0, 2))

        # parse overview and get squad links (including commented HTML)
        soup = BeautifulSoup(page.content(), "html.parser")
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        squad_links = []
        for c in comments:
            cs = BeautifulSoup(c, "html.parser")
            squad_links.extend(cs.select("table a[href*='/squads/']"))
        # fallback: direct links if any
        if not squad_links:
            squad_links = soup.select("table a[href*='/squads/']")

        squads = {}
        for a in squad_links:
            href = a["href"]
            base = urljoin("https://fbref.com", href)
            # convert to explicit 2024-2025 team-season page: insert '/2024-2025/' before last segment
            parts = base.rsplit("/", 1)
            season_url = f"{parts[0]}/2024-2025/{parts[1]}"
            squads[a.get_text(strip=True)] = season_url

        print(f"Found {len(squads)} squads for 2024-25.")

        # iterate teams
        for team_name, team_url in squads.items():
            safe_team = team_name.replace("/", "-")
            print(f"\n=== Team: {safe_team} ===")
            try:
                page.goto(team_url)
                time.sleep(4 + random.uniform(0, 2))
                team_soup = BeautifulSoup(page.content(), "html.parser")

                # Find stats_standard_9 (player list)
                table = team_soup.find("table", id="stats_standard_9")
                if not table:
                    # search in commented blocks
                    for c in team_soup.find_all(string=lambda text: isinstance(text, Comment)):
                        cs = BeautifulSoup(c, "html.parser")
                        table = cs.find("table", id="stats_standard_9")
                        if table:
                            break
                if not table:
                    print("No stats_standard_9 found for", team_name)
                    continue

                tbody = table.find("tbody")
                if not tbody:
                    print("No tbody in stats_standard_9 for", team_name)
                    continue

                os.makedirs(safe_team, exist_ok=True)

                # For each player row, find the player page and then the match logs link
                for tr in tbody.find_all("tr"):
                    if tr.get("class") and "thead" in tr.get("class"):
                        continue
                    player_th = tr.find("th", {"data-stat": "player"})
                    if not player_th:
                        # sometimes player cell is different; try first <th>
                        player_th = tr.find("th")
                    if not player_th:
                        continue
                    a = player_th.find("a", href=True)
                    if not a:
                        continue
                    player_name = player_th.get_text(strip=True).replace("/", "-")
                    player_profile_url = urljoin("https://fbref.com", a["href"])
                    print("  -> Player:", player_name)

                    # load player profile page
                    page.goto(player_profile_url)
                    time.sleep(3 + random.uniform(0, 2))
                    player_page_soup = BeautifulSoup(page.content(), "html.parser")

                    # 1) Try to find a direct matchlogs link
                    matchlog_link = find_matchlogs_link(player_page_soup)

                    # Force season 2024-2025 in match logs URL
                    if "/matchlogs/" in matchlog_link:
                        matchlog_link = re.sub(r"/matchlogs/\d{4}-\d{4}/", "/matchlogs/2024-2025/", matchlog_link)
                        # handle cases where no season is in URL (like /matchlogs/all_comps/)
                        if not re.search(r"/matchlogs/\d{4}-\d{4}/", matchlog_link):
                            matchlog_link = matchlog_link.replace("/matchlogs/", "/matchlogs/2024-2025/")

                        page.goto(matchlog_link)
                        time.sleep(3 + random.uniform(0, 2))
                        match_soup = BeautifulSoup(page.content(), "html.parser")
                        # collect tables from page and comments
                        all_tables = match_soup.find_all("table")
                        for c in match_soup.find_all(string=lambda text: isinstance(text, Comment)):
                            all_tables.extend(BeautifulSoup(c, "html.parser").find_all("table"))

                        chosen = choose_matchlog_table(all_tables)
                        if chosen:
                            rows = extract_table_rows(chosen)
                        else:
                            rows = []
                    else:
                        # 2) No dedicated link found — maybe match logs are embedded on the profile page (in comments)
                        all_tables = player_page_soup.find_all("table")
                        for c in player_page_soup.find_all(string=lambda text: isinstance(text, Comment)):
                            all_tables.extend(BeautifulSoup(c, "html.parser").find_all("table"))
                        chosen = choose_matchlog_table(all_tables)
                        rows = extract_table_rows(chosen) if chosen else []

                    # 3) Filter rows to Premier League 2024-2025 when possible
                    if rows:
                        # find header name that indicates competition
                        comp_field = None
                        if len(rows) > 0:
                            sample_keys = rows[0].keys()
                            for k in sample_keys:
                                lk = k.lower()
                                if "comp" in lk or "competition" in lk:
                                    comp_field = k
                                    break
                        if comp_field:
                            filtered = [r for r in rows if "premier" in (r.get(comp_field) or "").lower() and "2024" in str(r.get("season", "")) or True]
                            # if filtered non-empty and likely matches, use filtered
                            if any("premier" in (r.get(comp_field) or "").lower() for r in rows):
                                rows = [r for r in rows if "premier" in (r.get(comp_field) or "").lower()]
                        # If comp_field missing, try to infer from caption of chosen table (skip for now)

                    # Save rows to file (if any), else create empty list file
                    filename = os.path.join(safe_team, f"{player_name}.json")
                    with open(filename, "w", encoding="utf-8") as f:
                        json.dump(rows, f, indent=2, ensure_ascii=False)
                    print(f"    saved {filename} ({len(rows)} rows)")

                    # polite delay between players
                    time.sleep(random.uniform(1.5, 3.5))

                # bigger delay between teams
                time.sleep(random.uniform(6, 12))

            except Exception as e:
                print("Error scraping team", team_name, ":", e)
                continue

        browser.close()

if __name__ == "__main__":
    scrape_premier_league_players_matchlogs()
