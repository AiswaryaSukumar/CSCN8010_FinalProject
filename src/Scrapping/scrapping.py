from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://successportal.conestogac.on.ca"
START_URL = (
    "https://successportal.conestogac.on.ca/"
    "students/resources/search/?order=Relevance&topicsUseAnd=true"
)

OUTPUT_CSV = "success_portal_resources.csv"


def create_driver():
    # If chromedriver is on PATH, this is enough
    service = Service()
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")   # use headless later if you want
    driver = webdriver.Chrome(service=service, options=options)
    driver.maximize_window()
    return driver


def scrape_search_page_html(html):
    """
    Parse ONE results page (already loaded) and return list of dicts.
    Uses the classes from your screenshot.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Each resource card:
    # <div class="list-group-item resource-list-item resource-item-doc"> ... </div>
    cards = soup.select("div.list-group-item.resource-list-item")
    print(f"  Found {len(cards)} resources on this page")

    page_rows = []

    for card in cards:
        classes = card.get("class", [])
        # e.g. ['list-group-item', 'resource-list-item', 'resource-item-doc']
        res_type = None
        for c in classes:
            if c.startswith("resource-item-"):
                res_type = c.replace("resource-item-", "")  # 'doc', 'faq', 'link', ...

        # Title is inside:
        # <h4 class="list-group-item-heading">
        #    <a href="/students/docs/detail/...">Experience Profile</a>
        h4 = card.select_one("h4.list-group-item-heading")
        a = h4.select_one("a") if h4 else None

        title = a.get_text(strip=True) if a else ""
        href = a["href"] if (a and a.has_attr("href")) else ""
        if href and not href.startswith("http"):
            detail_url = BASE_URL + href
        else:
            detail_url = href

        # Summary:
        # <p class="list-group-item-text">Your experience matters! ...</p>
        p = card.select_one("p.list-group-item-text")
        summary = p.get_text(" ", strip=True) if p else ""

        page_rows.append(
            {
                "title": title,
                "summary": summary,
                "detail_url": detail_url,
                "resource_type": res_type,
            }
        )

    return page_rows


def get_total_pages(driver):
    """
    Look at the pagination bar and figure out how many pages exist.
    Uses the numeric buttons at the bottom (1 2 3 4 ...).
    """
    time.sleep(2)
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    # Typical structure: <ul class="pagination"><li><a>1</a></li> ... </ul>
    page_links = soup.select("ul.pagination li a")
    numbers = []
    for a in page_links:
        text = a.get_text(strip=True)
        if text.isdigit():
            numbers.append(int(text))

    if not numbers:
        return 1

    max_page = max(numbers)
    print(f"Detected {max_page} pages in total")
    return max_page


def main():
    driver = create_driver()

    try:
        print("Opening start URL...")
        driver.get(START_URL)

        # Let you manually log in (Conestoga SSO).
        input(
            "\nLog in in the browser window.\n"
            "When you see the resource list (like in your screenshot), "
            "press ENTER here to start scraping..."
        )

        # Detect how many pages we have
        total_pages = get_total_pages(driver)

        all_rows = []

        # Loop through all pages
        for page in range(1, total_pages + 1):
            print(f"\n=== Scraping page {page}/{total_pages} ===")

            if page != 1:
                # Click the correct page number in the pagination bar
                # (button text is just "2", "3", etc.)
                page_link = driver.find_element(By.LINK_TEXT, str(page))
                page_link.click()
                time.sleep(2)  # wait for Angular to load new results

            html = driver.page_source
            page_rows = scrape_search_page_html(html)
            all_rows.extend(page_rows)

        # Save everything
        df = pd.DataFrame(all_rows)
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"\nDone! Saved {len(df)} resources to {OUTPUT_CSV}")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
