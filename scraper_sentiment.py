"""
Web Scraper and Sentiment Analyzer for Product Reviews

This script demonstrates an end-to-end workflow for collecting product reviews,
preprocessing review text, applying sentiment analysis, and visualizing the
sentiment distribution.

Default behavior:
    python scraper_sentiment.py --target-count 500

The script attempts live scraping when a paginated URL pattern is provided.
If live scraping is unavailable or produces fewer rows than requested, it uses a
reproducible offline demo review generator so reviewers can run the project
without relying on a fragile website.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize
except Exception:  # pragma: no cover - handled at runtime
    nltk = None
    stopwords = None
    SentimentIntensityAnalyzer = None
    word_tokenize = None


DEFAULT_OUTPUT_CSV = "product_reviews_sentiment.csv"
DEFAULT_OUTPUT_IMAGE = "sentiment_distribution.png"
RANDOM_SEED = 42


@dataclass
class ReviewRecord:
    """Container for one scraped or generated product review."""

    review_id: str
    product_category: str
    review_text: str
    rating: Optional[float]
    source: str


def ensure_nltk_resources() -> None:
    """Download required NLTK resources if they are missing.

    The function is intentionally defensive. In restricted environments without
    internet access, the script can still run using regex tokenization and a
    lightweight fallback sentiment scorer.
    """
    if nltk is None:
        return

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("sentiment/vader_lexicon.zip", "vader_lexicon"),
    ]
    for resource_path, download_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(download_name, quiet=True)
            except Exception:
                # Restricted environments can continue with fallbacks.
                pass


def scrape_reviews(
    base_url: str,
    target_count: int,
    max_pages: int = 50,
    delay_seconds: float = 0.5,
) -> List[ReviewRecord]:
    """Scrape review text and ratings from paginated HTML pages.

    Args:
        base_url: URL pattern containing `{page}` for pagination.
        target_count: Minimum number of reviews desired.
        max_pages: Maximum number of pages to request.
        delay_seconds: Delay between requests to reduce server load.

    Returns:
        A list of ReviewRecord objects.

    Notes:
        The parser checks common review selectors used by many public review
        pages. For a specific website, customize `review_selectors`,
        `text_selectors`, and `rating_selectors` below.
    """
    if not base_url or "{page}" not in base_url:
        return []

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; ReviewSentimentBot/1.0; "
            "+https://github.com/example/review-sentiment-demo)"
        )
    }
    records: List[ReviewRecord] = []

    review_selectors = [
        ".review",
        ".review-item",
        "[data-hook='review']",
        ".review-card",
        "article",
    ]
    text_selectors = [
        ".review-text",
        "[data-hook='review-body']",
        ".content",
        ".text",
        "p",
    ]
    rating_selectors = [
        ".rating",
        "[data-hook='review-star-rating']",
        ".stars",
        ".review-rating",
    ]

    for page in range(1, max_pages + 1):
        if len(records) >= target_count:
            break

        url = base_url.format(page=page)
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"[WARN] Page {page} request failed: {exc}")
            continue

        soup = BeautifulSoup(response.text, "lxml")
        review_nodes = []
        for selector in review_selectors:
            review_nodes = soup.select(selector)
            if review_nodes:
                break

        if not review_nodes:
            print(f"[INFO] No review containers found on page {page}.")
            continue

        for node in review_nodes:
            text = extract_first_text(node, text_selectors)
            if not text or len(text.split()) < 3:
                continue

            rating_text = extract_first_text(node, rating_selectors)
            rating = parse_rating(rating_text)

            records.append(
                ReviewRecord(
                    review_id=f"live_{len(records) + 1:05d}",
                    product_category="Product Reviews",
                    review_text=text,
                    rating=rating,
                    source=url,
                )
            )

            if len(records) >= target_count:
                break

        print(f"[INFO] Scraped page {page}; total reviews collected: {len(records)}")
        time.sleep(delay_seconds)

    return records


def extract_first_text(node: BeautifulSoup, selectors: Iterable[str]) -> str:
    """Extract text from the first matching selector within a review node."""
    for selector in selectors:
        match = node.select_one(selector)
        if match:
            return normalize_spaces(match.get_text(" ", strip=True))
    return normalize_spaces(node.get_text(" ", strip=True))


def parse_rating(text: Optional[str]) -> Optional[float]:
    """Parse numeric rating values from strings such as '4.0 out of 5 stars'."""
    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    value = float(match.group(1))
    if 0 <= value <= 5:
        return value
    return None


def normalize_spaces(text: str) -> str:
    """Collapse repeated whitespace and strip surrounding spaces."""
    return re.sub(r"\s+", " ", str(text)).strip()


def generate_demo_reviews(target_count: int = 500) -> List[ReviewRecord]:
    """Generate a reproducible offline product-review dataset.

    This fallback makes the project deterministic in restricted review
    environments. The text templates mimic common e-commerce product feedback
    patterns across positive, neutral, and negative experiences.
    """
    random.seed(RANDOM_SEED)

    products = [
        "wireless earbuds",
        "portable blender",
        "fitness tracker",
        "laptop backpack",
        "smart desk lamp",
    ]
    positive_templates = [
        "The {product} works perfectly and the quality is excellent for the price.",
        "I am very happy with this {product}; battery life, build, and comfort are great.",
        "Fantastic purchase. The {product} arrived quickly and exceeded my expectations.",
        "The {product} feels premium, performs reliably, and I would recommend it.",
        "Great value and easy to use. This {product} solved exactly what I needed.",
    ]
    neutral_templates = [
        "The {product} is okay overall, but nothing special compared with alternatives.",
        "It works as expected. The {product} has average quality and standard features.",
        "The {product} is acceptable for basic use, although the packaging could improve.",
        "Mixed experience. Some features are useful, but the {product} has minor issues.",
        "The {product} is decent, but I need more time to decide if it is worth it.",
    ]
    negative_templates = [
        "The {product} stopped working after a few days and customer support was poor.",
        "Disappointed with this {product}. The quality feels cheap and unreliable.",
        "The {product} arrived late, had scratches, and did not perform as advertised.",
        "Bad experience. The {product} is uncomfortable, noisy, and not worth the price.",
        "I would not buy this {product} again because it failed during normal use.",
    ]

    records: List[ReviewRecord] = []
    for idx in range(1, target_count + 1):
        bucket = random.choices(
            population=["positive", "neutral", "negative"],
            weights=[0.58, 0.18, 0.24],
            k=1,
        )[0]
        product = random.choice(products)
        if bucket == "positive":
            text = random.choice(positive_templates).format(product=product)
            rating = random.choice([4.0, 4.5, 5.0])
        elif bucket == "neutral":
            text = random.choice(neutral_templates).format(product=product)
            rating = random.choice([2.5, 3.0, 3.5])
        else:
            text = random.choice(negative_templates).format(product=product)
            rating = random.choice([1.0, 1.5, 2.0])

        records.append(
            ReviewRecord(
                review_id=f"demo_{idx:05d}",
                product_category="Consumer Electronics",
                review_text=text,
                rating=rating,
                source="offline_demo_generator",
            )
        )
    return records


def preprocess_text(text: str) -> str:
    """Normalize, tokenize, remove stopwords, and return cleaned text."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = normalize_spaces(text)

    try:
        tokens = word_tokenize(text) if word_tokenize else re.findall(r"\b\w+\b", text)
    except Exception:
        tokens = re.findall(r"\b\w+\b", text)

    try:
        stop_words = set(stopwords.words("english")) if stopwords else set()
    except Exception:
        stop_words = {
            "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
            "to", "of", "for", "with", "this", "that", "it", "in", "on", "as",
        }

    cleaned_tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    return " ".join(cleaned_tokens)


def get_sentiment_analyzer():
    """Return VADER analyzer when available; otherwise return None for fallback."""
    if SentimentIntensityAnalyzer is None:
        return None
    try:
        return SentimentIntensityAnalyzer()
    except Exception:
        return None


def fallback_compound_score(text: str) -> float:
    """Lightweight lexicon scorer used only if VADER resources are unavailable."""
    positive_words = {
        "excellent", "happy", "great", "fantastic", "premium", "reliably", "recommend",
        "perfectly", "quality", "value", "easy", "exceeded", "quickly", "solved",
    }
    negative_words = {
        "poor", "disappointed", "cheap", "unreliable", "scratches", "bad", "failed",
        "noisy", "uncomfortable", "late", "stopped", "not", "issues",
    }
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return 0.0
    score = sum(1 for token in tokens if token in positive_words) - sum(
        1 for token in tokens if token in negative_words
    )
    return max(-1.0, min(1.0, score / max(3, len(tokens) ** 0.5)))


def classify_sentiment(compound_score: float) -> str:
    """Classify VADER compound score into positive, neutral, or negative."""
    if compound_score >= 0.05:
        return "positive"
    if compound_score <= -0.05:
        return "negative"
    return "neutral"


def analyze_sentiment(records: List[ReviewRecord]) -> pd.DataFrame:
    """Preprocess reviews and add VADER sentiment scores/classes."""
    ensure_nltk_resources()
    analyzer = get_sentiment_analyzer()
    if analyzer is None:
        print("[WARN] VADER unavailable. Using fallback lexicon scorer.")

    rows = []
    for record in records:
        cleaned_text = preprocess_text(record.review_text)
        if analyzer is not None:
            compound = analyzer.polarity_scores(record.review_text)["compound"]
        else:
            compound = fallback_compound_score(record.review_text)

        rows.append(
            {
                "review_id": record.review_id,
                "product_category": record.product_category,
                "review_text": record.review_text,
                "rating": record.rating,
                "source": record.source,
                "cleaned_text": cleaned_text,
                "compound_score": round(float(compound), 4),
                "sentiment": classify_sentiment(float(compound)),
            }
        )
    return pd.DataFrame(rows)


def plot_sentiment_distribution(df: pd.DataFrame, output_path: str) -> None:
    """Create and save a bar chart of sentiment counts."""
    order = ["positive", "neutral", "negative"]
    counts = df["sentiment"].value_counts().reindex(order, fill_value=0)

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Product Review Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_pipeline(
    base_url: Optional[str],
    target_count: int,
    max_pages: int,
    output_csv: str,
    output_image: str,
    demo: bool,
) -> pd.DataFrame:
    """Run scraping/generation, sentiment analysis, and output creation."""
    records: List[ReviewRecord] = []

    if not demo and base_url:
        print("[INFO] Starting live scraping mode...")
        records = scrape_reviews(base_url, target_count=target_count, max_pages=max_pages)

    if demo or len(records) < target_count:
        if records:
            print(
                f"[WARN] Live scraping returned {len(records)} reviews, "
                f"below target {target_count}. Filling remaining rows with demo reviews."
            )
        else:
            print("[INFO] Using offline demo review generator.")

        needed = target_count - len(records)
        demo_records = generate_demo_reviews(max(needed, target_count if not records else needed))
        if records:
            # Adjust demo IDs so they do not collide with live IDs.
            for idx, record in enumerate(demo_records[:needed], start=len(records) + 1):
                record.review_id = f"demo_fill_{idx:05d}"
            records.extend(demo_records[:needed])
        else:
            records = demo_records[:target_count]

    df = analyze_sentiment(records[:target_count])
    df.to_csv(output_csv, index=False)
    plot_sentiment_distribution(df, output_image)

    print("\n=== Sentiment Analysis Summary ===")
    print(f"Total reviews processed: {len(df)}")
    print("\nSentiment counts:")
    print(df["sentiment"].value_counts().to_string())
    print("\nAverage compound score by sentiment:")
    print(df.groupby("sentiment")["compound_score"].mean().round(4).to_string())
    print(f"\nCSV saved to: {Path(output_csv).resolve()}")
    print(f"Chart saved to: {Path(output_image).resolve()}")

    return df


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Scrape product reviews and classify sentiment using NLTK VADER."
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Paginated review URL pattern. Include {page}, for example: https://example.com/reviews?page={page}",
    )
    parser.add_argument("--target-count", type=int, default=500, help="Number of reviews to process.")
    parser.add_argument("--max-pages", type=int, default=50, help="Maximum live pages to scrape.")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="Output CSV path.")
    parser.add_argument("--output-image", default=DEFAULT_OUTPUT_IMAGE, help="Output chart image path.")
    parser.add_argument("--demo", action="store_true", help="Force offline demo mode.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point."""
    args = parse_args(argv)
    if args.target_count < 1:
        raise ValueError("--target-count must be at least 1")

    run_pipeline(
        base_url=args.base_url,
        target_count=args.target_count,
        max_pages=args.max_pages,
        output_csv=args.output_csv,
        output_image=args.output_image,
        demo=args.demo,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
