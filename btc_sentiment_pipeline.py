"""
BTC Sentiment Pipeline — Twitter/X via tweety-ns (free)
=========================================================

SETUP
─────
  1. pip install tweety-ns vaderSentiment
  2. python setup_session.py --cookie YOUR_AUTH_TOKEN
  3. python diagnose.py
  4. python btc_sentiment_pipeline.py

USAGE AS A MODULE
─────────────────
  from btc_sentiment_pipeline import SentimentEngine
  import asyncio

  async def main():
      engine = SentimentEngine()
      await engine.start()
      signal = engine.get_signal()
      print(signal.to_dict())

  asyncio.run(main())
"""

import re
import time
import asyncio
import threading
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean, stdev

from tweety import TwitterAsync
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

SESSION_NAME = "btc_session"

TWITTER_SEARCH_QUERIES = [
    "bitcoin",
    "btc",
    "$BTC",
    "#bitcoin",
]
POLL_INTERVAL_SECONDS = 30
WINDOW_MINUTES = 15
MAX_TWEETS_PER_QUERY = 50
MIN_TEXT_LENGTH = 10
ENGAGEMENT_WEIGHT_ENABLED = True

CRYPTO_LEXICON = {
    "moon": 2.0, "mooning": 2.5, "to the moon": 2.5,
    "rekt": -3.0, "rekted": -3.0, "wrecked": -2.5,
    "ngmi": -2.0, "wagmi": 2.0,
    "hodl": 1.0, "hodling": 1.0,
    "dump": -2.0, "dumping": -2.5, "dumped": -2.0,
    "pump": 1.5, "pumping": 2.0, "pumped": 1.5,
    "rugpull": -3.5, "rug": -3.0, "rugged": -3.0,
    "bullish": 2.0, "bearish": -2.0,
    "capitulation": -2.5, "capitulating": -2.5,
    "fud": -1.5, "fudding": -1.5,
    "dip": -1.0, "buying the dip": 1.5, "btfd": 1.5,
    "ath": 2.0, "all time high": 2.0,
    "crash": -3.0, "crashing": -3.0,
    "scam": -2.5, "ponzi": -3.0,
    "send it": 1.5, "lfg": 2.0, "lets go": 1.5,
    "gm": 0.5, "gn": 0.3,
    "paper hands": -1.5, "diamond hands": 1.5,
    "bag holder": -2.0, "bagholder": -2.0,
    "short": -1.0, "shorting": -1.5,
    "long": 1.0, "longing": 1.0,
    "liquidated": -2.5, "liquidation": -2.0,
    "100k": 1.5, "1m": 2.0,
    "bottom": -1.0, "bottoming": -0.5,
    "top": -0.5, "topping": -1.0,
    "reversal": 0.5, "breakout": 1.5, "breakdown": -1.5,
    "support": 0.5, "resistance": -0.3,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sentiment")


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class ScoredText:
    text: str
    compound: float
    positive: float
    negative: float
    neutral: float
    engagement: int
    timestamp: float
    text_id: str


@dataclass
class SentimentSignal:
    score: float
    velocity: float
    volume: int
    bullish_ratio: float
    bearish_ratio: float
    std_dev: float
    window_minutes: int
    timestamp: str

    def to_dict(self) -> dict:
        return self.__dict__


# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"^RT\s+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"pic\.twitter\.com/\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# CORE ENGINE
# ─────────────────────────────────────────────

class SentimentEngine:
    def __init__(self, session_name: str = SESSION_NAME):
        self.session_name = session_name

        self.analyzer = SentimentIntensityAnalyzer()
        self.analyzer.lexicon.update(CRYPTO_LEXICON)
        log.info(f"VADER loaded with {len(CRYPTO_LEXICON)} crypto terms")

        self.window: deque[ScoredText] = deque()
        self.seen_ids: set[str] = set()
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        self._stats = {
            "polls_total": 0,
            "polls_empty": 0,
            "tweets_total": 0,
            "last_poll_time": None,
            "last_poll_ms": 0,
            "last_tweet_count": 0,
            "started_at": None,
        }

    def _score_text(self, text: str) -> dict:
        scores = self.analyzer.polarity_scores(text)
        return {
            "compound": scores["compound"],
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
        }

    async def _fetch_tweets(self, app: TwitterAsync) -> list[ScoredText]:
        results = []
        cutoff = time.time() - (WINDOW_MINUTES * 60)

        for query in TWITTER_SEARCH_QUERIES:
            try:
                search_results = await app.search(query)

                for tweet in search_results:
                    tid = str(tweet.id)
                    if tid in self.seen_ids:
                        continue
                    self.seen_ids.add(tid)

                    tweet_time = tweet.date.timestamp() if tweet.date else time.time()
                    if tweet_time < cutoff:
                        continue

                    raw_text = tweet.text or ""
                    text = clean_text(raw_text)
                    if len(text) < MIN_TEXT_LENGTH:
                        continue

                    engagement = (
                        (tweet.likes or 0) +
                        (tweet.retweet_counts or 0) * 2 +
                        (tweet.reply_counts or 0)
                    )

                    scores = self._score_text(text)
                    results.append(ScoredText(
                        text=text[:200],
                        compound=scores["compound"],
                        positive=scores["positive"],
                        negative=scores["negative"],
                        neutral=scores["neutral"],
                        engagement=max(engagement, 1),
                        timestamp=tweet_time,
                        text_id=tid,
                    ))

            except Exception as e:
                log.warning(f"Search error for '{query}': {e}")

        return results

    def _evict_old(self):
        cutoff = time.time() - (WINDOW_MINUTES * 60)
        while self.window and self.window[0].timestamp < cutoff:
            old = self.window.popleft()
            self.seen_ids.discard(old.text_id)

    def get_signal(self) -> SentimentSignal:
        """Sub-millisecond — reads from in-memory deque."""
        with self._lock:
            self._evict_old()
            entries = list(self.window)

        now_iso = datetime.now(timezone.utc).isoformat()

        if not entries:
            return SentimentSignal(
                score=0.0, velocity=0.0, volume=0,
                bullish_ratio=0.5, bearish_ratio=0.5,
                std_dev=0.0, window_minutes=WINDOW_MINUTES,
                timestamp=now_iso,
            )

        if ENGAGEMENT_WEIGHT_ENABLED:
            total_weight = sum(e.engagement for e in entries)
            if total_weight > 0:
                score = sum(e.compound * e.engagement for e in entries) / total_weight
            else:
                score = mean(e.compound for e in entries)
        else:
            score = mean(e.compound for e in entries)

        n = len(entries)
        bullish = sum(1 for e in entries if e.compound > 0.05) / n
        bearish = sum(1 for e in entries if e.compound < -0.05) / n

        compounds = [e.compound for e in entries]
        sd = stdev(compounds) if len(compounds) > 1 else 0.0

        mid = n // 2
        if mid > 0:
            first_half = mean(e.compound for e in entries[:mid])
            second_half = mean(e.compound for e in entries[mid:])
            elapsed_min = (entries[-1].timestamp - entries[0].timestamp) / 60
            velocity = (second_half - first_half) / max(elapsed_min, 0.5)
        else:
            velocity = 0.0

        return SentimentSignal(
            score=round(score, 4),
            velocity=round(velocity, 4),
            volume=n,
            bullish_ratio=round(bullish, 4),
            bearish_ratio=round(bearish, 4),
            std_dev=round(sd, 4),
            window_minutes=WINDOW_MINUTES,
            timestamp=now_iso,
        )

    async def _poll_loop_async(self):
        app = TwitterAsync(self.session_name)

        log.info(f"Loading session from {self.session_name}.tw_session ...")
        try:
            await app.connect()
            log.info(f"Logged in as: {app.me}")
        except Exception as e:
            log.error(
                f"Failed to load session: {e}\n"
                f"Run 'python setup_session.py --cookie YOUR_TOKEN' first."
            )
            self._running = False
            return

        log.info("Polling loop started")
        while self._running:
            t0 = time.perf_counter()

            new_tweets = await self._fetch_tweets(app)
            new_tweets.sort(key=lambda x: x.timestamp)

            with self._lock:
                self.window.extend(new_tweets)
                self._evict_old()

            elapsed = (time.perf_counter() - t0) * 1000
            count = len(new_tweets)

            self._stats["polls_total"] += 1
            self._stats["last_poll_time"] = datetime.now(timezone.utc).isoformat()
            self._stats["last_poll_ms"] = round(elapsed)
            self._stats["last_tweet_count"] = count
            self._stats["tweets_total"] += count

            if count == 0:
                self._stats["polls_empty"] += 1

            log.info(
                f"Poll #{self._stats['polls_total']}: "
                f"{count} tweets in {elapsed:.0f}ms | "
                f"Window: {len(self.window)}"
            )

            if count == 0:
                log.warning(
                    f"⚠ EMPTY POLL — 0 tweets. "
                    f"Empty: {self._stats['polls_empty']}/{self._stats['polls_total']}. "
                    f"Run 'python diagnose.py' to troubleshoot."
                )
            else:
                signal = self.get_signal()
                log.info(
                    f"Signal → score={signal.score:+.3f}  "
                    f"vel={signal.velocity:+.4f}/min  "
                    f"vol={signal.volume}  "
                    f"bull={signal.bullish_ratio:.0%} bear={signal.bearish_ratio:.0%}  "
                    f"σ={signal.std_dev:.3f}"
                )

            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    def _run_async_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._poll_loop_async())

    async def start(self):
        if self._running:
            return
        self._running = True
        self._stats["started_at"] = datetime.now(timezone.utc).isoformat()
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()
        log.info("SentimentEngine started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        log.info("SentimentEngine stopped")

    def health_check(self) -> dict:
        with self._lock:
            window_size = len(self.window)

        issues = []
        status = "healthy"

        if not self._running:
            issues.append("Engine not running")
            status = "unhealthy"

        if self._stats["polls_total"] == 0:
            issues.append("No polls yet — just started")
            status = "starting"
        elif self._stats["polls_total"] > 3:
            empty_rate = self._stats["polls_empty"] / self._stats["polls_total"]
            if empty_rate == 1.0:
                issues.append("ALL polls empty — session may be expired")
                status = "unhealthy"
            elif empty_rate > 0.5:
                issues.append(f"High empty rate: {empty_rate:.0%}")
                status = "degraded"

        if window_size == 0 and self._stats["polls_total"] > 2:
            issues.append("Window empty")
            if status != "unhealthy":
                status = "degraded"

        return {
            "status": status,
            "issues": issues,
            "running": self._running,
            "window_size": window_size,
            "polls": self._stats,
        }


# ─────────────────────────────────────────────
# STANDALONE RUNNER
# ─────────────────────────────────────────────

async def main():
    engine = SentimentEngine()
    await engine.start()
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        log.info("Shutting down...")
        engine.stop()


if __name__ == "__main__":
    asyncio.run(main())