# Install first:
# pip install playwright bs4 pandas
# playwright install

import os
import time
import random
from bs4 import BeautifulSoup, Comment
import pandas as pd
from playwright.sync_api import sync_playwright

def scrape_premier_league_2024_25():
    with sync_playwright() as p:
        # Launch Playwright browser
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        # Premier League 2024-25 overview page
        league_url = "https://fbref.com/en/comps/9/2024-2025/2024-2025-Premier-League-Stats"
        page.goto(league_url)
        time.sleep(5 + random.uniform(0, 3))

        html = page.content()
        soup = BeautifulSoup(html, "html.parser")

        # Extract squad links from HTML comments
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        squad_links = []
        for c in comments:
            comment_soup = BeautifulSoup(c, "html.parser")
            links = comment_soup.select("table a[href*='/squads/']")
            if links:
                squad_links.extend(links)

        squads = {a.text: "https://fbref.com" + a["href"] for a in squad_links}
        print(f"Found {len(squads)} squads.")

        # Function to scrape all tables for a team
        def scrape_team(team_name, team_url):
            page.goto(team_url)
            time.sleep(5 + random.uniform(0, 3))

            html = page.content()
            soup = BeautifulSoup(html, "html.parser")

            # Extract tables including those in comments
            all_tables = soup.find_all("table")
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for c in comments:
                comment_soup = BeautifulSoup(c, "html.parser")
                all_tables.extend(comment_soup.find_all("table"))

            # Create folder for team
            os.makedirs(team_name, exist_ok=True)

            # Extract tables
            for table in all_tables:
                table_id = table.get("id") or "unknown_table"
                thead = table.find("thead")
                tbody = table.find("tbody")
                if not thead or not tbody:
                    continue

                headers = [th.get_text(strip=True) for th in thead.find_all("tr")[-1].find_all("th")]

                rows = []
                for tr in tbody.find_all("tr"):
                    if tr.get("class") and "thead" in tr.get("class"):
                        continue
                    cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
                    if cells:
                        rows.append(cells)

                if rows:
                    filename = f"{team_name}/{table_id}_2024-25.csv"
                    pd.DataFrame(rows, columns=headers).to_csv(filename, index=False)
                    print(f"Saved {filename}")

        # Loop through all squads
        for team_name, team_url in squads.items():
            safe_name = team_name.replace("/", "-")
            print(f"\nScraping {safe_name} ...")
            parts = team_url.rsplit("/", 1)
            team_url = f"{parts[0]}/2024-2025/{parts[1]}"
            scrape_team(safe_name, team_url)
            time.sleep(random.uniform(5, 10))  # human-like delay

        browser.close()

# Run scraper
scrape_premier_league_2024_25()
