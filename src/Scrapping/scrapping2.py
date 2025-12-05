import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://orientation.conestogac.on.ca/questions/faq"
OUTPUT_CSV = "orientation_faq.csv"


def scrape_orientation_faq(url):
    print(f"Fetching {url} ...")

    # Add a browser-like User-Agent to bypass bot-blocking
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Try multiple methods to locate FAQ content
    main = soup.find("main")
    if not main:
        print("WARNING: <main> not found — trying body instead.")
        main = soup.find("body")

    if not main:
        raise RuntimeError("Cannot find FAQ section on page")

    # All FAQs are <details> elements
    faq_items = main.find_all("details")
    print(f"Found {len(faq_items)} FAQ entries")

    if len(faq_items) == 0:
        print("The page HTML received:")
        print(resp.text[:1500])
        raise RuntimeError("No <details> found — the website may still be blocking scraping.")

    rows = []

    for i, details in enumerate(faq_items, start=1):
        # QUESTION part
        summary_tag = details.find("summary")
        question = summary_tag.get_text(" ", strip=True) if summary_tag else ""

        # ANSWER part (all other children inside <details>)
        answer_parts = []
        for child in details.find_all(recursive=False):
            if child.name == "summary":
                continue
            text = child.get_text(" ", strip=True)
            if text:
                answer_parts.append(text)

        answer = " ".join(answer_parts)

        rows.append({
            "question": question,
            "answer": answer,
            "source_url": url
        })

        print(f"[{i}] Scraped: {question[:70]}...")

    return rows


def main():
    rows = scrape_orientation_faq(URL)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} FAQ items to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
