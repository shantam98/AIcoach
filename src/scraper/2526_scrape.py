# Install first:
# pip install playwright bs4 pandas
# playwright install

import os
import time
import random
from bs4 import BeautifulSoup, Comment
import pandas as pd
from playwright.sync_api import sync_playwright

def scrape_fbref_premier_league():
    with sync_playwright() as p:
        # Launch browser (headless=False to avoid bot detection)
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        # Step 1: Premier League overview page
        league_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
        page.goto(league_url)
        time.sleep(5)

        html = page.content()
        soup = BeautifulSoup(html, "html.parser")

        # Step 2: Extract squad links from HTML comments
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        squad_links = []
        for c in comments:
            comment_soup = BeautifulSoup(c, "html.parser")
            links = comment_soup.select("table a[href*='/squads/']")
            if links:
                squad_links.extend(links)

        squads = {a.text: "https://fbref.com" + a["href"] for a in squad_links}
        print(f"Found {len(squads)} squads.")

        # Step 3: Function to scrape all tables from a squad page
        def scrape_team(team_name, team_url):
            page.goto(team_url)
            time.sleep(5 + random.uniform(0, 3))  # human-like delay

            html = page.content()
            soup = BeautifulSoup(html, "html.parser")

            # Include tables inside comments
            all_tables = soup.find_all("table")
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for c in comments:
                comment_soup = BeautifulSoup(c, "html.parser")
                all_tables.extend(comment_soup.find_all("table"))

            # Create folder for team
            os.makedirs(team_name, exist_ok=True)

            # Extract each table
            for table in all_tables:
                table_id = table.get("id") or "unknown_table"

                thead = table.find("thead")
                if not thead:
                    print(f"Skipping table {table_id} (no thead)")
                    continue  # skip tables without headers

                # Headers: last row in thead
                header_rows = thead.find_all("tr")
                headers = [th.get_text(strip=True) for th in header_rows[-1].find_all("th")]

                # Rows
                tbody = table.find("tbody")
                if not tbody:
                    print(f"Skipping table {table_id} (no tbody)")
                    continue
                rows = []
                for tr in tbody.find_all("tr"):
                    if tr.get("class") and "thead" in tr.get("class"):
                        continue
                    cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
                    if cells:
                        rows.append(cells)
                if rows:
                    df = pd.DataFrame(rows, columns=headers)
                    filename = f"{team_name}/{table_id}.csv"
                    df.to_csv(filename, index=False)
                    print(f"Saved {filename}")
                else:
                    print(f"No data rows found for table {table_id}")

        # Step 4: Loop through all squads
        for team_name, team_url in squads.items():
            safe_name = team_name.replace("/", "-")
            print(f"\nScraping {safe_name} ...")
            scrape_team(safe_name, team_url)
            time.sleep(random.uniform(5, 10))  # human-like delay

        browser.close()

# Run the scraper
scrape_fbref_premier_league()
