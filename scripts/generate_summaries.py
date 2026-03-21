#!/usr/bin/env python3
"""Fetch RSS feeds for SITAWARE columns, generate AI summaries and relevance flags."""

import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import quote

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-haiku-4-5-20251001"
SUMMARIES_FILE = "summaries.json"

COLUMNS = {
    "monrovia": {
        "label": "Monrovia",
        "relevance_filter": True,
        "feeds": [
            {"name": "Monrovia Now", "url": "https://www.monrovianow.com/feeds/posts/default?alt=rss"},
            {"name": "Google News", "url": "https://news.google.com/rss/search?q=%22Monrovia%22+%22California%22+-Liberia+when:7d&hl=en-US&gl=US&ceid=US:en"},
            {"name": "Patch Monrovia", "url": "https://patch.com/feeds/aol/california/monrovia"},
            {"name": "HeySoCal", "url": "https://heysocal.com/category/neighborhood/san-gabriel-valley/monroviaweekly/feed/"},
        ],
    },
    "sgv": {
        "label": "San Gabriel Valley",
        "relevance_filter": True,
        "feeds": [
            {"name": "Google News SGV", "url": "https://news.google.com/rss/search?q=%22San+Gabriel+Valley%22+when:7d&hl=en-US&gl=US&ceid=US:en"},
            {"name": "Pasadena Now", "url": "https://pasadenanow.com/main/feed"},
            {"name": "Colorado Blvd", "url": "https://www.coloradoboulevard.net/feed/"},
            {"name": "ABC7", "url": "https://abc7.com/feed/"},
        ],
    },
    "county": {
        "label": "Los Angeles County",
        "relevance_filter": False,
        "feeds": [
            {"name": "LAist", "url": "https://laist.com/index.atom"},
            {"name": "Fox 11 LA", "url": "https://www.foxla.com/rss/category/news"},
            {"name": "MyNewsLA", "url": "https://mynewsla.com/feed/"},
            {"name": "Google News LA", "url": "https://news.google.com/rss/search?q=%22Los+Angeles%22+County+-sports+-entertainment+when:3d&hl=en-US&gl=US&ceid=US:en"},
        ],
    },
}


def fetch_feed(url, timeout=15):
    """Fetch and parse an RSS/Atom feed, return list of headline strings."""
    headlines = []
    try:
        req = Request(url, headers={"User-Agent": "SITAWARE-Bot/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        root = ET.fromstring(data)
        # RSS 2.0
        for item in root.findall(".//item"):
            title_el = item.find("title")
            if title_el is not None and title_el.text:
                headlines.append(title_el.text.strip())
        # Atom
        if not headlines:
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall(".//atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                if title_el is not None and title_el.text:
                    headlines.append(title_el.text.strip())
        # Atom without namespace
        if not headlines:
            for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
                title_el = entry.find("{http://www.w3.org/2005/Atom}title")
                if title_el is not None and title_el.text:
                    headlines.append(title_el.text.strip())
    except Exception as e:
        print(f"  Warning: failed to fetch {url}: {e}", file=sys.stderr)
    return headlines[:20]  # Cap at 20 per feed


def call_anthropic(prompt):
    """Call the Anthropic API and return the text response."""
    import json as _json
    url = "https://api.anthropic.com/v1/messages"
    body = _json.dumps({
        "model": MODEL,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
    })
    with urlopen(req, timeout=30) as resp:
        result = _json.loads(resp.read())
    return result["content"][0]["text"]


def generate_summary(column_label, headlines):
    """Generate 3-4 bullet points ranked by importance for a column."""
    if not headlines:
        return "- No recent headlines available for this briefing."
    headline_list = "\n".join(f"- {h}" for h in headlines[:30])
    prompt = f"""You are a local news briefing writer for {column_label} in Southern California. Below are today's headlines from local news feeds.

Write exactly 3 or 4 bullet points summarizing the most important stories, ranked by importance (most important first). Each bullet must be 12 words or fewer. Headline style, not full sentences. Clear, punchy, scannable. No em dashes. No filler. Focus on stories relevant to {column_label}.

Headlines:
{headline_list}

Respond with ONLY the bullet points, one per line, each starting with "- ". Nothing else."""
    return call_anthropic(prompt)


def classify_relevance(column_label, headlines):
    """Classify each headline as relevant or not for the given area."""
    if not headlines:
        return {}
    headline_list = "\n".join(f"{i}: {h}" for i, h in enumerate(headlines[:40]))
    prompt = f"""You are classifying news headlines for a local dashboard focused on {column_label}, Southern California.

For each headline below, decide if it is actually relevant to {column_label} specifically. Mark headlines as IRRELEVANT if they are:
- National or international news not specific to {column_label}
- Statewide California news not specific to this area
- Sports or entertainment news
- Stories about other regions that just happened to appear in a local feed

Respond with ONLY a JSON object mapping the headline index number to true (relevant) or false (irrelevant). Example: {{"0": true, "1": false, "2": true}}

Headlines:
{headline_list}"""
    text = call_anthropic(prompt)
    # Extract JSON from response
    try:
        # Find the JSON object in the response
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        # If parsing fails, assume all relevant
        print(f"  Warning: could not parse relevance flags for {column_label}", file=sys.stderr)
        return {str(i): True for i in range(len(headlines))}


def get_briefing_label():
    """Return the briefing label based on current Pacific time."""
    from datetime import timedelta
    # Approximate Pacific time (UTC-7 for PDT, UTC-8 for PST)
    # For simplicity, check month for DST
    utc_now = datetime.now(timezone.utc)
    month = utc_now.month
    # PDT: March-November, PST: November-March (approximate)
    offset = timedelta(hours=-7) if 3 <= month <= 10 else timedelta(hours=-8)
    pacific = utc_now + offset
    hour = pacific.hour
    if hour < 10:
        return "8 AM"
    elif hour < 15:
        return "12 PM"
    else:
        return "5 PM"


def main():
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Load existing summaries as fallback
    existing = {}
    if os.path.exists(SUMMARIES_FILE):
        try:
            with open(SUMMARIES_FILE) as f:
                existing = json.load(f)
        except Exception:
            pass

    now = datetime.now(timezone.utc).isoformat()
    briefing_label = get_briefing_label()
    output = {
        "generated": now,
        "briefing": briefing_label,
        "columns": {},
    }

    for col_key, col_config in COLUMNS.items():
        print(f"Processing {col_config['label']}...")
        all_headlines = []
        for feed in col_config["feeds"]:
            print(f"  Fetching {feed['name']}...")
            headlines = fetch_feed(feed["url"])
            print(f"    Got {len(headlines)} headlines")
            for h in headlines:
                all_headlines.append({"headline": h, "source": feed["name"]})

        if not all_headlines:
            print(f"  No headlines fetched, keeping previous summary")
            if col_key in existing.get("columns", {}):
                output["columns"][col_key] = existing["columns"][col_key]
            else:
                output["columns"][col_key] = {
                    "summary": "No headlines available.",
                    "headlines": [],
                    "relevance": {},
                }
            continue

        # Generate summary
        print(f"  Generating summary...")
        try:
            headline_texts = [h["headline"] for h in all_headlines]
            summary = generate_summary(col_config["label"], headline_texts)
            print(f"  Summary: {summary[:80]}...")
        except Exception as e:
            print(f"  Error generating summary: {e}", file=sys.stderr)
            summary = existing.get("columns", {}).get(col_key, {}).get(
                "summary", "Summary temporarily unavailable."
            )

        # Classify relevance (only for Monrovia and SGV)
        relevance = {}
        if col_config["relevance_filter"]:
            print(f"  Classifying relevance...")
            try:
                headline_texts = [h["headline"] for h in all_headlines]
                relevance = classify_relevance(col_config["label"], headline_texts)
                irrelevant_count = sum(1 for v in relevance.values() if not v)
                print(f"  Flagged {irrelevant_count}/{len(relevance)} as irrelevant")
            except Exception as e:
                print(f"  Error classifying relevance: {e}", file=sys.stderr)

        output["columns"][col_key] = {
            "summary": summary,
            "headlines": all_headlines,
            "relevance": relevance,
        }

    # Write output
    with open(SUMMARIES_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {SUMMARIES_FILE} at {now}")
    print(f"Briefing: {briefing_label}")


if __name__ == "__main__":
    main()
