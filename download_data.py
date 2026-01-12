import os
import glob
import re
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader

# --- CONFIG ---
# SEC requires a user-agent in this format: "Name email@domain.com"
# Since this is a test, we use a placeholder for the email.
USER_AGENT = "TestUser user123@company.com"
DATA_DIR = "data"
RAW_DIR = "temp_sec_raw"


def extract_readable_content_from_xbrl(html_content):
    """
    Extract readable text from XBRL inline HTML document.
    """
    try:
        # Parse HTML content
        soup = BeautifulSoup(html_content, 'lxml')

        # Remove XBRL tags but keep their text content
        for tag in soup.find_all(True):
            if tag.name and ':' in tag.name:  # XBRL namespace tags like ix:nonNumeric
                tag.unwrap()

        # Remove script, style, and other non-content tags
        for tag in soup(["script", "style", "head", "title", "meta", "link"]):
            tag.decompose()

        # Get text content
        text = soup.get_text(separator=" ")

        # Clean up the text
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            # Skip empty lines and lines that are mostly XBRL artifacts
            if (line and
                len(line) > 5 and
                not line.startswith('<?xml') and
                not line.startswith('<!--') and
                not re.match(r'^[{}\[\],\s]*$', line) and
                not line.startswith('http://') and
                'xmlns:' not in line):
                cleaned_lines.append(line)

        # Join lines and clean up extra whitespace
        text = '\n'.join(cleaned_lines)
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double newline

        return text.strip()

    except Exception as e:
        print(f"Error extracting content: {e}")
        return ""


def find_html_content_in_sec_filing(sec_content):
    """
    Find and extract the main HTML document from SEC filing.
    """
    # Look for the main 10-K document with HTML content
    lines = sec_content.split('\n')

    for i, line in enumerate(lines):
        # Look for the start of an HTML document
        if line.strip() == '<TEXT>' and i > 0:
            # Check if this section contains HTML
            start_idx = i + 1
            html_start = None

            # Look ahead to see if this contains HTML
            for j in range(start_idx, min(start_idx + 20, len(lines))):
                if '<html' in lines[j].lower():
                    html_start = j
                    break

            if html_start is not None:
                # Find the end of this TEXT section
                end_idx = start_idx
                for j in range(start_idx, len(lines)):
                    if lines[j].strip() == '</TEXT>':
                        end_idx = j
                        break

                # Extract the HTML content
                html_lines = lines[html_start:end_idx]
                html_content = '\n'.join(html_lines)

                # Check if this looks like the main document (longer content)
                if len(html_content) > 10000:  # Reasonable size threshold
                    return html_content

    return None


def download_and_process():
    # 1. Initialize Downloader
    dl = Downloader("TestUser", "user123@company.com", RAW_DIR)

    companies = ["AAPL"] # , "MSFT", "GOOG", "AMZN", "TSLA", "META", "NFLX", "NVDA", "JPM", "JNJ"]
    # AAPL (Apple) - Technology/Consumer Electronics
    # MSFT (Microsoft) - Technology/Software
    # GOOG (Google/Alphabet) - Technology/Internet
    # AMZN (Amazon) - E-commerce/Cloud Services
    # TSLA (Tesla) - Automotive/Clean Energy
    # META (Meta/Facebook) - Social Media/Technology
    # NFLX (Netflix) - Entertainment/Streaming
    # NVDA (NVIDIA) - Semiconductors/AI
    # JPM (JPMorgan Chase) - Financial Services/Banking
    # JNJ (Johnson & Johnson) - Healthcare/Pharmaceuticals

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    limit = 5
    print(f"--- Downloading {len(companies)*limit} Filings ---")

    for ticker in companies:
        print(f"Fetching {ticker}...")
        # Get last 5 10-Ks
        dl.get("10-K", ticker, limit=limit)

    print("\n--- Processing & Cleaning ---")

    # 2. Walk through the raw download folder to find the full-submission.txt documents
    # Structure: temp_sec_raw/sec-edgar-filings/TICKER/10-K/FILING_ID/full-submission.txt
    files = glob.glob(f"{RAW_DIR}/sec-edgar-filings/*/10-K/*/full-submission.txt")

    count = 0
    for filepath in files:
        try:
            # Extract metadata from path
            parts = filepath.split(os.sep)
            ticker = parts[-4]  # e.g. AAPL
            filing_id = parts[-2]  # e.g. 0000320193-21-000105

            print(f"Processing {ticker} - {filing_id}...")

            # Read Raw SEC submission file
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                raw_content = f.read()

            # Extract HTML content from SEC filing
            html_content = find_html_content_in_sec_filing(raw_content)

            if not html_content:
                print(f"Warning: No HTML content found for {ticker}")
                continue

            # Extract readable text from HTML/XBRL
            text_content = extract_readable_content_from_xbrl(html_content)

            if len(text_content) < 1000:  # Too short, probably extraction failed
                print(f"Warning: Extracted content too short for {ticker}")
                continue

            # Save to clean data folder
            # Use ticker and filing ID for filename
            filename = f"{ticker}_10K_{filing_id}.txt"
            out_path = os.path.join(DATA_DIR, filename)

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text_content)

            print(f"Saved: {filename} ({len(text_content):,} characters)")
            count += 1

        except Exception as e:
            print(f"Skipping {filepath}: {e}")

    print(f"\n-> Successfully processed {count} documents into '{DATA_DIR}/'")


if __name__ == "__main__":
    download_and_process()
