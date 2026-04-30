from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from analyzer.data import get_news


POSITIVE_WORDS = {
    "beat", "beats", "surge", "upgrade", "growth", "bullish", "profit",
    "record", "strong", "outperform", "rebound", "expand", "optimism",
}
NEGATIVE_WORDS = {
    "miss", "misses", "cut", "cuts", "downgrade", "bearish", "loss",
    "weak", "lawsuit", "probe", "fall", "falls", "drop", "warning",
    "recession", "inflation", "tariff",
}


@dataclass
class NewsItem:
    title: str
    publisher: str
    sentiment_label: str
    sentiment_score: int
    published_at: str
    link: str


def _score_text(text: str) -> int:
    words = {w.strip(".,:;!?()[]{}").lower() for w in text.split()}
    return sum(1 for w in words if w in POSITIVE_WORDS) - sum(1 for w in words if w in NEGATIVE_WORDS)


def _label(score: int) -> str:
    if score >= 2:
        return "bullish"
    if score <= -2:
        return "bearish"
    return "neutral"


def fetch_news(symbol: str, max_items: int = 6) -> list[NewsItem]:
    items = get_news(symbol)

    out: list[NewsItem] = []
    for item in items[:max_items]:
        content = item.get("content") or {}
        title = content.get("title") or item.get("title") or ""
        publisher = content.get("provider", {}).get("displayName") or item.get("publisher") or "Unknown"
        link = content.get("canonicalUrl", {}).get("url") or item.get("link") or ""
        ts = content.get("pubDate") or item.get("providerPublishTime")
        published = ""
        try:
            if isinstance(ts, str):
                published = ts[:16].replace("T", " ")
            elif ts:
                published = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
        except Exception:
            published = ""
        score = _score_text(title)
        out.append(
            NewsItem(
                title=title,
                publisher=publisher,
                sentiment_label=_label(score),
                sentiment_score=score,
                published_at=published,
                link=link,
            )
        )
    return out


def aggregate_sentiment(items: list[NewsItem]) -> tuple[str, float]:
    if not items:
        return "neutral", 0.0
    avg = sum(i.sentiment_score for i in items) / len(items)
    if avg >= 1.0:
        return "bullish", avg
    if avg <= -1.0:
        return "bearish", avg
    return "neutral", avg
