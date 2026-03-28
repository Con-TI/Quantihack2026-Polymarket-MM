"""
BTC Sentiment Pipeline — Diagnostics (tweety-ns)
==================================================
Tests each component before running the pipeline.

Usage:  python diagnose.py
"""

import sys
import os
import time
import asyncio
from datetime import datetime, timezone

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def ok(msg):     print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg):   print(f"  {RED}✗{RESET} {msg}")
def warn(msg):   print(f"  {YELLOW}⚠{RESET} {msg}")
def info(msg):   print(f"  {CYAN}ℹ{RESET} {msg}")
def header(msg): print(f"\n{BOLD}{'─'*50}\n  {msg}\n{'─'*50}{RESET}")


# ─────────────────────────────────────────────
# 1. DEPENDENCIES
# ─────────────────────────────────────────────

def check_dependencies():
    header("1. PYTHON DEPENDENCIES")

    required = {
        "tweety": "tweety-ns — Twitter scraper",
        "vaderSentiment": "VADER — sentiment scoring",
    }

    all_ok = True
    for pkg, desc in required.items():
        try:
            __import__(pkg)
            ok(f"{pkg} — {desc}")
        except ImportError:
            fail(f"{pkg} — {desc} — NOT INSTALLED")
            fix = "tweety-ns" if pkg == "tweety" else pkg
            info(f"  Fix: pip install {fix}")
            all_ok = False

    return all_ok


# ─────────────────────────────────────────────
# 2. SESSION FILE
# ─────────────────────────────────────────────

def check_session():
    header("2. SESSION FILE")

    session_file = "btc_session.tw_session"

    if os.path.exists(session_file):
        size = os.path.getsize(session_file)
        age_hours = (time.time() - os.path.getmtime(session_file)) / 3600
        ok(f"Found: {session_file} ({size} bytes, {age_hours:.1f}h old)")
        if age_hours > 168:
            warn("Session is >7 days old — may have expired")
        return True
    else:
        fail(f"NOT FOUND: {session_file}")
        info("Create one with: python setup_session.py --cookie YOUR_AUTH_TOKEN")
        return False


# ─────────────────────────────────────────────
# 3. TWITTER CONNECTION
# ─────────────────────────────────────────────

async def test_twitter():
    header("3. TWITTER CONNECTION & SEARCH")

    from tweety import TwitterAsync

    app = TwitterAsync("btc_session")

    try:
        await app.connect()
        ok(f"Logged in as: {app.me}")
    except Exception as e:
        fail(f"Session load failed: {e}")
        info("Re-run: python setup_session.py --cookie YOUR_AUTH_TOKEN")
        return False

    queries = ["bitcoin", "$BTC"]
    total = 0

    for query in queries:
        try:
            t0 = time.perf_counter()
            results = await app.search(query)
            elapsed = (time.perf_counter() - t0) * 1000

            count = 0
            sample = None
            for tweet in results:
                count += 1
                if sample is None:
                    sample = tweet
                if count >= 10:
                    break

            if count > 0:
                ok(f"Query '{query}': {count}+ tweets in {elapsed:.0f}ms")
                if sample and sample.date:
                    age_min = (time.time() - sample.date.timestamp()) / 60
                    info(f"  Latest: {age_min:.0f} min ago")
                    info(f"  Text: \"{(sample.text or '')[:80]}\"")
                total += count
            else:
                warn(f"Query '{query}': 0 tweets in {elapsed:.0f}ms")

        except Exception as e:
            fail(f"Query '{query}' failed: {e}")

    print()
    if total > 0:
        ok(f"Twitter working — {total}+ tweets found")
        return True
    else:
        fail("0 tweets returned — session may be expired")
        return False


# ─────────────────────────────────────────────
# 4. VADER SCORING
# ─────────────────────────────────────────────

def test_vader():
    header("4. VADER SENTIMENT SCORING")

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    try:
        from btc_sentiment_pipeline import CRYPTO_LEXICON
        analyzer.lexicon.update(CRYPTO_LEXICON)
        ok(f"Crypto lexicon: {len(CRYPTO_LEXICON)} terms")
    except ImportError:
        warn("Could not load CRYPTO_LEXICON from btc_sentiment_pipeline.py")

    tests = [
        ("Bitcoin to the moon! LFG!",            "bullish"),
        ("BTC got rekt, total capitulation",      "bearish"),
        ("I bought 0.5 BTC yesterday",            "~neutral"),
        ("This is a rugpull, get out now!",        "strong bearish"),
        ("Diamond hands, we're not selling",       "bullish"),
    ]

    print()
    for text, expected in tests:
        c = analyzer.polarity_scores(text)["compound"]
        if c > 0.05:
            label = f"{GREEN}BULL {c:+.3f}{RESET}"
        elif c < -0.05:
            label = f"{RED}BEAR {c:+.3f}{RESET}"
        else:
            label = f"{YELLOW}NEUT {c:+.3f}{RESET}"
        print(f"    {label}  \"{text[:45]}\" → expected: {expected}")

    print()
    ok("VADER working")
    return True


# ─────────────────────────────────────────────
# 5. FULL PIPELINE
# ─────────────────────────────────────────────

async def test_pipeline():
    header("5. FULL PIPELINE TEST")

    from btc_sentiment_pipeline import SentimentEngine
    from tweety import TwitterAsync

    engine = SentimentEngine()
    app = TwitterAsync("btc_session")

    try:
        await app.connect()
    except Exception as e:
        fail(f"Session failed: {e}")
        return False

    info("Fetching tweets...")
    t0 = time.perf_counter()
    tweets = await engine._fetch_tweets(app)
    elapsed = (time.perf_counter() - t0) * 1000

    info(f"{len(tweets)} tweets in {elapsed:.0f}ms")

    if not tweets:
        fail("No tweets fetched")
        return False

    engine.window.extend(tweets)
    signal = engine.get_signal()

    print()
    ok("Signal computed:")
    info(f"  Score:    {signal.score:+.4f}")
    info(f"  Velocity: {signal.velocity:+.4f}/min")
    info(f"  Volume:   {signal.volume}")
    info(f"  Bullish:  {signal.bullish_ratio:.0%}")
    info(f"  Bearish:  {signal.bearish_ratio:.0%}")

    print()
    info("Sample tweets:")
    for t in tweets[:5]:
        icon = "🟢" if t.compound > 0.05 else ("🔴" if t.compound < -0.05 else "⚪")
        print(f"    {icon} {t.compound:+.3f}  \"{t.text[:55]}\"")

    print()
    ok(f"Pipeline working — {len(tweets)} tweets scored")
    return True


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

async def main():
    print(f"\n{BOLD}BTC Sentiment Pipeline — Diagnostics{RESET}")
    print(f"{'='*50}")
    print(f"Time:   {datetime.now(timezone.utc).isoformat()}")
    print(f"Python: {sys.version.split()[0]}")

    results = {}

    results["dependencies"] = check_dependencies()
    if not results["dependencies"]:
        print(f"\n{RED}Install dependencies first.{RESET}")
        sys.exit(1)

    results["session"] = check_session()
    if not results["session"]:
        print(f"\n{RED}Create a session first.{RESET}")
        sys.exit(1)

    results["twitter"] = await test_twitter()
    results["vader"] = test_vader()

    if results["twitter"]:
        results["pipeline"] = await test_pipeline()
    else:
        results["pipeline"] = False

    header("SUMMARY")
    for name, passed in results.items():
        (ok if passed else fail)(f"{name}: {'PASS' if passed else 'FAIL'}")

    if all(results.values()):
        print(f"\n{GREEN}All passed! Run: python btc_sentiment_pipeline.py{RESET}\n")
    else:
        print(f"\n{RED}Fix issues above and re-run.{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())