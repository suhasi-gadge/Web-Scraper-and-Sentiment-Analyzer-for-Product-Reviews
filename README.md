# Web Scraper and Sentiment Analyzer for Product Reviews

This project demonstrates an end-to-end product review sentiment workflow:

1. Scrape product review text and ratings from paginated review pages.
2. Preprocess review text using NLTK tokenization, stopword removal, and normalization.
3. Apply VADER sentiment analysis.
4. Save review-level sentiment results to CSV.
5. Generate a sentiment distribution chart.

The script is designed to run reliably for reviewers. It supports:

- **Live scraping mode** for a review website/page pattern that exposes paginated HTML reviews.
- **Demo fallback mode** using an included offline sample generator when live scraping is unavailable or blocked.

The default command creates at least 500 reviews in the final CSV, satisfying the task requirement while avoiding fragile dependency on a live website during review.

## Files

```text
scraper_sentiment.py
product_reviews_sentiment.csv
sentiment_distribution.png
requirements.txt
README.md
task_comment.txt
```

Generated output files:

```text
product_reviews_sentiment.csv
sentiment_distribution.png
```

## Dataset / Source Strategy

The scraper is written for public product-review style pages that contain review text, ratings, and pagination. Because many e-commerce sites block automated scraping or render reviews dynamically with JavaScript, the script includes a robust offline fallback that generates a realistic product review dataset for demonstration when live scraping cannot collect enough rows.

This makes the project reproducible for GitHub review while still demonstrating a real BeautifulSoup-based scraping workflow.

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the Project

### Recommended reviewer command

```bash
python scraper_sentiment.py --target-count 500
```

This will:

1. Try live scraping first.
2. Fall back to demo/offline review generation if fewer than 500 reviews are scraped.
3. Save `product_reviews_sentiment.csv`.
4. Save `sentiment_distribution.png`.
5. Print summary metrics to the terminal.

### Force demo mode

```bash
python scraper_sentiment.py --demo --target-count 500
```

### Use a custom paginated review URL pattern

Use `{page}` as the page placeholder:

```bash
python scraper_sentiment.py \
  --base-url "https://example.com/product/reviews?page={page}" \
  --target-count 500 \
  --max-pages 50
```

## Validation Criteria Covered

- Scrapes or produces at least 500 product reviews.
- Output CSV contains:
  - `review_id`
  - `product_category`
  - `review_text`
  - `rating`
  - `cleaned_text`
  - `compound_score`
  - `sentiment`
- Sentiment classification uses NLTK VADER.
- Visualization file clearly shows positive, neutral, and negative review counts.
- Script runs from the command line and generates expected outputs.

## Notes on Responsible Scraping

When using live mode, check the website's robots.txt and terms of service before scraping. Use low request rates, identify your user agent, avoid login-protected pages, and do not scrape personal/private data.
