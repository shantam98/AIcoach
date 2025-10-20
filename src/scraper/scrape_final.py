from playwright.sync_api import sync_playwright
import time
from bs4 import BeautifulSoup, Comment

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # run visible browser
    context = browser.new_context()
    page = context.new_page()
    
    page.goto("https://fbref.com/en/comps/9/Premier-League-Stats")
    time.sleep(5)  # wait for page to fully load
    
    # Scroll down to simulate human behavior
    page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
    time.sleep(2)
    
    html = page.content()
    soup = BeautifulSoup(html, "html.parser")
    
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    squad_links = []
    for c in comments:
        comment_soup = BeautifulSoup(c, "html.parser")
        links = comment_soup.select("table a[href*='/squads/']")
        if links:
            squad_links.extend(links)
    
    squads = {a.text: "https://fbref.com" + a["href"] for a in squad_links}
    print(f"Found {len(squads)} squads")
    
    browser.close()
