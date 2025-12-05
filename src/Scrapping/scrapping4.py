import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://www.conestogac.on.ca/student-rights/faq"
OUTPUT_CSV = "student_rights_faq.csv"


def scrape_faq_page(url):
    print(f"Fetching {url} ...")

    # Pretend to be a browser (avoids simple bot blocking)
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

    main = soup.find("main")
    if not main:
        print("WARNING: <main> not found, trying <body> instead")
        main = soup.find("body")
    if not main:
        raise RuntimeError("Cannot find FAQ section on the page")

    # all FAQ entries
    faq_items = main.find_all("details")
    print(f"Found {len(faq_items)} FAQ entries")

    if not faq_items:
        print(resp.text[:1500])
        raise RuntimeError("No <details> found – page structure changed or blocked")

    rows = []

    for i, details in enumerate(faq_items, start=1):
        # question
        summary_tag = details.find("summary")
        question = summary_tag.get_text(" ", strip=True) if summary_tag else ""

        # answer (all children except <summary>)
        answer_parts = []
        for child in details.find_all(recursive=False):
            if child.name == "summary":
                continue
            text = child.get_text(" ", strip=True)
            if text:
                answer_parts.append(text)

        answer = " ".join(answer_parts)

        rows.append(
            {
                "question": question,
                "answer": answer,
                "source_url": url,
            }
        )

        print(f"[{i}] {question[:70]}...")

    return rows


def main():
    rows = scrape_faq_page(URL)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} FAQs to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
