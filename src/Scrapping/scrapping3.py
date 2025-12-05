import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://orientation.conestogac.on.ca/questions/faq"
OUTPUT_CSV = "orientation_faq.csv"


def scrape_orientation_faq(url):
    print(f"Fetching {url} ...")
    resp = requests.get(url)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # The FAQ content is inside <main>
    main = soup.find("main")
    if not main:
        raise RuntimeError("Could not find <main> on the page")

    # All FAQ entries are <details> blocks
    faq_items = main.find_all("details")
    print(f"Found {len(faq_items)} FAQ entries")

    rows = []

    for i, details in enumerate(faq_items, start=1):

        # --- QUESTION ---
        summary_tag = details.find("summary")
        question = summary_tag.get_text(" ", strip=True) if summary_tag else ""

        # --- ANSWER ---
        # All non-summary tags inside <details> form the answer
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
    print(f"\nSaved {len(df)} FAQs to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
