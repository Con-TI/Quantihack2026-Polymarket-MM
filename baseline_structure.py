"""
baseline_structure.py — Market-making framework for Polymarket 15-minute binary options.

Multi-asset combined controller with:
  - Microprice logistic-regression alpha signal
  - Cross-asset correlation-adjusted exposure
  - Trade intensity imbalance signal
  - Asymmetric inventory skew

Manages Up/Down books for multiple underlyings (BTC, XRP, ETH, SOL) in a
single Controller instance.  Books are keyed by (underlying, outcome) tuples.

Signal integration:
  Signals adjust EFFECTIVE EXPOSURE (not quote positions directly) so
  the asymmetric skew fill model continues to work:

    microprice_pred = 2*sigmoid(slope*ema + intercept) - 1   -> [-1,+1]
    intensity_sig   = (bullish_ema - bearish_ema) / total    -> [-1,+1]
    directional     = mp_weight*microprice + int_weight*intensity

    corr_exp        = own_net_exp + sum(corr_ij * other_net_exp_j)
    effective_exp   = corr_exp - directional
    scaled          = direction * effective_exp
    -> asymmetric skew as before
"""

import numpy as np
from typing import Dict, Callable, Optional, Tuple
import heapq
from scipy.stats import gaussian_kde, poisson
import pandas as pd
from collections import defaultdict


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------

class TradingModel:
    """
    Base class for alpha signal generation.
    Maintains state across messages; return None to skip alpha update.
    """
    def __init__(self):
        pass

    def run(self, msg) -> Optional[float]:
        """Return fair-value offset from mid (alpha), or None if message is irrelevant."""
        raise NotImplementedError


class TradeMsgSender:
    """Converts Controller orders into exchange submissions."""
    def __init__(self):
        pass

    def send(self, msg) -> int:
        """Submit order. msg = (price, quantity, side). Returns order_id."""
        raise NotImplementedError


class Logger:
    def __init__(self):
        pass

    def log_msg(self, msg):
        pass


# ---------------------------------------------------------------------------
# Per-book state
# ---------------------------------------------------------------------------

class BookState:
    """Order-book state for a single (underlying, outcome) pair."""

    def __init__(self, underlying: str, outcome: str, queue_pos_func: Callable):
        self.underlying = underlying
        self.outcome = outcome
        self.watchers: Dict[int, "QueueWatcher"] = {
            p: QueueWatcher(p, queue_pos_func) for p in range(1, 100)
        }
        self.best_bid: int = 49
        self.best_ask: int = 51
        self.mid_price: float = 50.0
        self.inventory: float = 0.0
        self.alpha_val: float = 0.0
        self.queue_sizes: Dict[int, float] = {i: 0.0 for i in range(1, 100)}
        self._book_initialized: bool = False

        # Signal state
        self.microprice_dev_ema: float = 0.0
        self.prev_mid_price: float = 0.0


# ---------------------------------------------------------------------------
# Core controller — multi-asset with signals
# ---------------------------------------------------------------------------

class Controller:
    """
    Market-making controller for complementary binary contracts across
    multiple underlyings.

    On each message it:
      1. Routes to the correct (underlying, outcome) book.
      2. Updates market state, microprice EMA, trade intensity, correlation.
      3. Processes fills gated by queue position.
      4. Computes reference bid/ask using signal-adjusted effective exposure.
      5. Cancels stale orders; replenishes empty levels.
    """

    def __init__(
        self,
        alpha_function: Callable,
        sender: TradeMsgSender,
        logger: Logger,
        queue_bound: float = 0,
        inventory_limit: float = 10,
        level_range: int = 4,
        wealth: float = 10000.0,
        queue_pos_func: Callable = lambda x: x,
        live: bool = True,
        skew_per_unit: float = 0.5,
        window_ms: int = 15 * 60 * 1000,
        queue_entry_frac: float = 0.5,
        # Signal parameters
        microprice_coefficients: Optional[Dict[str, Tuple[float, float]]] = None,
        microprice_ema_alpha: float = 0.3,
        intensity_ema_alpha: float = 0.05,
        correlation_window: int = 50,
        microprice_weight: float = 1.0,
        intensity_weight: float = 0.5,
        trend_ema_alpha: float = 0.05,
        trend_weight: float = 2.0,
        closeout_ms: int = 60 * 1000,
    ):
        self.wealth = wealth
        self._initial_wealth = wealth
        self.live = live

        self.books: Dict[Tuple[str, str], BookState] = {}
        self._underlyings: set = set()
        self._queue_pos_func = queue_pos_func

        self.alpha_function = alpha_function
        self.inventory_limit = inventory_limit
        self.queue_bound = queue_bound
        self.level_range = level_range
        self.sender = sender
        self.logger = logger

        self.skew_per_unit = skew_per_unit
        self.window_ms = window_ms
        self._current_ts: float = 0.0
        self.queue_entry_frac = queue_entry_frac

        # Cancel callback — set by live engine for exchange cancellation
        self.on_cancel: Optional[Callable[[int], None]] = None

        # Microprice signal
        self._mp_coeffs = microprice_coefficients or {}
        self._mp_ema_alpha = microprice_ema_alpha
        self._mp_weight = microprice_weight

        # Trade intensity signal — per underlying
        self._int_state: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"bull": 0.0, "bear": 0.0}
        )
        self._int_ema_alpha = intensity_ema_alpha
        self._int_weight = intensity_weight

        # Trend signal — EMA of Up book mid, deviation = directional signal
        self._trend_ema: Dict[str, float] = {}  # underlying -> EMA of mid
        self._trend_ema_alpha = trend_ema_alpha
        self._trend_weight = trend_weight

        # End-of-window closeout
        self._closeout_ms = closeout_ms  # stop new orders this many ms before expiry

        # Cross-asset correlation
        self._corr_window = correlation_window
        self._mid_returns: Dict[str, list] = defaultdict(list)
        self._corr_matrix: Dict[Tuple[str, str], float] = {}
        self._corr_counter = 0
        self._corr_interval = 10

    # ------------------------------------------------------------------
    # Book management
    # ------------------------------------------------------------------

    def add_book(self, underlying: str, outcome: str):
        key = (underlying, outcome)
        self.books[key] = BookState(underlying, outcome, self._queue_pos_func)
        self._underlyings.add(underlying)

    def net_exposure(self, underlying: str) -> float:
        """Net directional exposure for one asset: positive = bullish."""
        up = self.books.get((underlying, "Up"))
        down = self.books.get((underlying, "Down"))
        return (up.inventory if up else 0.0) - (down.inventory if down else 0.0)

    @staticmethod
    def _direction(outcome: str) -> float:
        return 1.0 if outcome == "Up" else -1.0

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _update_microprice(self, book: BookState):
        bb, ba = book.best_bid, book.best_ask
        if not (0 < bb < ba < 100):
            return
        v_bid = book.queue_sizes.get(bb, 0.0)
        v_ask = book.queue_sizes.get(ba, 0.0)
        total = v_bid + v_ask
        if total < 1e-6:
            return
        microprice = (v_bid * ba + v_ask * bb) / total
        dev = microprice - (bb + ba) / 2.0
        # For Down book, high imbalance is bearish → flip sign
        if book.outcome == "Down":
            dev = -dev
        a = self._mp_ema_alpha
        book.microprice_dev_ema = a * dev + (1.0 - a) * book.microprice_dev_ema

    def _microprice_prediction(self, underlying: str) -> float:
        """Logistic regression P(uptick) mapped to [-1, +1]."""
        coeffs = self._mp_coeffs.get(underlying)
        if coeffs is None:
            return 0.0
        slope, intercept = coeffs
        # Use Up book EMA; fall back to Down book (already sign-flipped)
        up = self.books.get((underlying, "Up"))
        down = self.books.get((underlying, "Down"))
        book = up if (up and up._book_initialized) else down
        if book is None or not book._book_initialized:
            return 0.0
        z = intercept + slope * book.microprice_dev_ema
        p_up = 1.0 / (1.0 + np.exp(-z))
        return 2.0 * p_up - 1.0  # map [0,1] → [-1,+1]

    def _update_intensity(self, underlying: str, outcome: str, side, size: float):
        a = self._int_ema_alpha
        st = self._int_state[underlying]
        is_bullish = (outcome == "Up" and side == 1) or (outcome == "Down" and side == 0)
        # Decay both, add impulse weighted by size
        st["bull"] = (1.0 - a) * st["bull"] + (a * size if is_bullish else 0.0)
        st["bear"] = (1.0 - a) * st["bear"] + (0.0 if is_bullish else a * size)

    def _intensity_signal(self, underlying: str) -> float:
        st = self._int_state[underlying]
        total = st["bull"] + st["bear"]
        if total < 1e-9:
            return 0.0
        return (st["bull"] - st["bear"]) / total

    def _update_trend(self, book: BookState):
        """Update EMA of Up book mid price. Trend = EMA - mid, positive = bullish."""
        if book.outcome != "Up":
            return
        ul = book.underlying
        a = self._trend_ema_alpha
        if ul not in self._trend_ema:
            self._trend_ema[ul] = book.mid_price
        else:
            self._trend_ema[ul] = a * book.mid_price + (1.0 - a) * self._trend_ema[ul]

    def _trend_signal(self, underlying: str) -> float:
        """EMA deviation from current mid, normalized. Positive = bullish trend."""
        up = self.books.get((underlying, "Up"))
        if up is None or not up._book_initialized or underlying not in self._trend_ema:
            return 0.0
        # EMA > mid means price has been rising → bullish
        # Normalize by spread so signal is in reasonable units
        spread = max(up.best_ask - up.best_bid, 1)
        return (self._trend_ema[underlying] - up.mid_price) / spread

    def _update_mid_returns(self, book: BookState):
        if book.outcome != "Up":
            return
        if book.prev_mid_price > 0:
            ret = book.mid_price - book.prev_mid_price
            self._mid_returns[book.underlying].append(ret)
            if len(self._mid_returns[book.underlying]) > self._corr_window * 2:
                self._mid_returns[book.underlying] = (
                    self._mid_returns[book.underlying][-self._corr_window:]
                )
            self._corr_counter += 1
            if self._corr_counter >= self._corr_interval:
                self._recompute_correlations()
                self._corr_counter = 0
        book.prev_mid_price = book.mid_price

    def _recompute_correlations(self):
        uls = [u for u in self._underlyings if len(self._mid_returns[u]) >= 10]
        if len(uls) < 2:
            return
        min_len = min(len(self._mid_returns[u]) for u in uls)
        arrays = {u: np.array(self._mid_returns[u][-min_len:]) for u in uls}
        for i, u1 in enumerate(uls):
            for u2 in uls[i + 1:]:
                if np.std(arrays[u1]) < 1e-12 or np.std(arrays[u2]) < 1e-12:
                    c = 0.0
                else:
                    c = float(np.corrcoef(arrays[u1], arrays[u2])[0, 1])
                    if np.isnan(c):
                        c = 0.0
                self._corr_matrix[(u1, u2)] = c
                self._corr_matrix[(u2, u1)] = c

    def correlated_exposure(self, underlying: str) -> float:
        own = self.net_exposure(underlying)
        for other in self._underlyings:
            if other != underlying:
                own += self._corr_matrix.get((underlying, other), 0.0) * self.net_exposure(other)
        return own

    def _effective_scaled(self, underlying: str, outcome: str) -> float:
        """Combine all signals into a single scaled value for skew/limits."""
        direction = self._direction(outcome)
        mp = self._microprice_prediction(underlying)
        it = self._intensity_signal(underlying)
        tr = self._trend_signal(underlying)
        target = (
            self._mp_weight * mp
            + self._int_weight * it
            + self._trend_weight * tr
        )
        effective = self.correlated_exposure(underlying) - target
        return direction * effective

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def parse(self, msg):
        data = msg.get("data", {})
        underlying = data.get("underlying")
        outcome = data.get("outcome")
        key = (underlying, outcome)
        if key not in self.books:
            return

        book = self.books[key]

        if not self.live:
            self._process_message(msg, book)
        else:
            self._process_message_live(msg, book)

        if not book._book_initialized:
            return

        time_remaining = self.window_ms - self._current_ts
        in_closeout = time_remaining <= self._closeout_ms

        if in_closeout:
            self._closeout_parse(book, underlying, outcome)
        else:
            self._normal_parse(book, underlying, outcome)

    def _normal_parse(self, book: BookState, underlying: str, outcome: str):
        """Standard market-making: signals + asymmetric skew."""
        scaled = self._effective_scaled(underlying, outcome)

        time_frac = (
            min(self._current_ts / self.window_ms, 1.0)
            if self.window_ms > 0
            else 0.0
        )
        time_multiplier = 1.0 + 1.0 * time_frac
        skew_amount = abs(scaled) * self.skew_per_unit * time_multiplier

        alpha = book.alpha_val
        if scaled > 0:
            reference_bid = int(np.clip(book.best_bid + alpha - skew_amount, 1, 98))
            reference_ask = int(np.clip(book.best_ask + alpha, 2, 99))
        elif scaled < 0:
            reference_bid = int(np.clip(book.best_bid + alpha, 1, 98))
            reference_ask = int(np.clip(book.best_ask + alpha + skew_amount, 2, 99))
        else:
            reference_bid = int(np.clip(book.best_bid + alpha, 1, 98))
            reference_ask = int(np.clip(book.best_ask + alpha, 2, 99))

        if reference_ask <= reference_bid:
            reference_ask = reference_bid + 1

        self._cancel_stale_orders(book, reference_bid, reference_ask, scaled)
        self._place_orders(book, reference_bid, reference_ask, scaled)

    def _closeout_parse(self, book: BookState, underlying: str, outcome: str):
        """
        Last-minute closeout: flatten net directional exposure to zero while
        minimizing cost.  Inventory on either side is fine as long as
        net_exposure(underlying) == 0.

        Strategy: cancel all risk-adding orders, keep only orders that
        reduce net exposure toward zero.  No new risk-adding orders.
        """
        direction = self._direction(outcome)
        net_exp = self.net_exposure(underlying)

        # Cancel ALL orders on this book first
        for p in range(1, 100):
            if book.watchers[p].orders:
                for oid in list(book.watchers[p].orders):
                    if self.on_cancel:
                        self.on_cancel(oid)
                    book.watchers[p].remove_order(oid)

        # If net exposure is already zero, don't place anything
        if abs(net_exp) < 0.5:
            return

        # Only place orders that reduce net_exposure toward 0.
        # net_exp > 0 (net long): sell Up (asks on Up) or buy Down (bids on Down)
        # net_exp < 0 (net short): buy Up (bids on Up) or sell Down (asks on Down)
        place_bids = False
        place_asks = False

        if net_exp > 0:
            # Need to reduce: sell Up or buy Down
            if outcome == "Up":
                place_asks = True   # sell Up
            else:
                place_bids = True   # buy Down
        else:
            # Need to increase: buy Up or sell Down
            if outcome == "Up":
                place_bids = True   # buy Up
            else:
                place_asks = True   # sell Down

        # Place at best price (aggressive) to maximize fill probability
        for i in range(self.level_range):
            if place_bids:
                bid_price = book.best_bid - i
                if bid_price > 0 and not book.watchers[bid_price].orders:
                    entry_pos = book.queue_sizes.get(bid_price, 0) * self.queue_entry_frac
                    oid = self.sender.send((bid_price, 1, 1, book.underlying, book.outcome))
                    book.watchers[bid_price].add_order(oid, 1, entry_pos)
            if place_asks:
                ask_price = book.best_ask + i
                if ask_price < 100 and not book.watchers[ask_price].orders:
                    entry_pos = book.queue_sizes.get(ask_price, 0) * self.queue_entry_frac
                    oid = self.sender.send((ask_price, 1, 0, book.underlying, book.outcome))
                    book.watchers[ask_price].add_order(oid, 1, entry_pos)

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def settle(self, settlement_prices: Dict[str, float]):
        """
        Settle all books.  settlement_prices maps underlying -> Up settlement
        price.  Down settles at 100 - up_price.
        """
        for (underlying, outcome), book in self.books.items():
            up_price = settlement_prices.get(underlying, 50.0)
            price = up_price if outcome == "Up" else 100.0 - up_price
            pnl = book.inventory * price
            self.wealth += pnl
            self.logger.log_msg({"type": "settlement", "data": {
                "underlying": underlying,
                "outcome": outcome,
                "settlement_price": price,
                "inventory_settled": book.inventory,
                "pnl_from_inventory": pnl,
                "final_wealth": self.wealth,
            }})
            book.inventory = 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_message(self, msg, book: BookState):
        t = msg["type"]

        ts = msg.get("data", {}).get("timestamp", msg.get("timestamp", 0))
        if ts:
            self._current_ts = float(ts)

        if t == "book":
            sizes = msg["data"]["sizes"]
            book.queue_sizes = {i: abs(sizes[i]) for i in range(1, 100)}
            book._book_initialized = True
            self._update_microprice(book)

        elif t == "price_change":
            d = msg["data"]
            book.queue_sizes[d["price"]] = max(
                book.queue_sizes[d["price"]] + d["size"], 0
            )
            book.watchers[d["price"]].update_queue_pos(
                order_quantity=d["size"],
                queue_size=book.queue_sizes[d["price"]],
                is_trade=False,
            )
            book.best_bid = d["best_bid"]
            book.best_ask = d["best_ask"]
            book.mid_price = (book.best_bid + book.best_ask) / 2.0
            self._update_microprice(book)
            self._update_mid_returns(book)
            self._update_trend(book)

        elif t == "last_trade":
            d = msg["data"]
            self._update_intensity(
                book.underlying, book.outcome, d["side"], float(d["size"])
            )
            self._handle_trade(d, book)

        out = self.alpha_function(msg)
        if out is not None:
            book.alpha_val = float(out)

    def _process_message_live(self, msg, book: BookState):
        """Parse raw Polymarket websocket messages into Controller state."""
        t = msg["type"]
        raw = msg["data"]
        ts = int(raw.get("timestamp", 0))
        if ts:
            self._current_ts = float(ts)

        if t == "book":
            sizes = [0.0] * 101
            for item in raw.get("bids", []):
                sizes[int(float(item["price"]) * 100)] = float(item["size"])
            for item in raw.get("asks", []):
                sizes[int(float(item["price"]) * 100)] = -float(item["size"])
            book.queue_sizes = {i: abs(sizes[i]) for i in range(1, 100)}
            # Derive best_bid / best_ask from the snapshot
            bids = [p for p in range(1, 100) if sizes[p] > 0]
            asks = [p for p in range(1, 100) if sizes[p] < 0]
            if bids:
                book.best_bid = max(bids)
            if asks:
                book.best_ask = min(asks)
            if bids or asks:
                book.mid_price = (book.best_bid + book.best_ask) / 2.0
            book._book_initialized = True
            self._update_microprice(book)
            self._update_mid_returns(book)
            self._update_trend(book)

        elif t == "price_change":
            price = int(float(raw["price"]) * 100)
            size = float(raw["size"])
            best_bid = int(float(raw["best_bid"]) * 100)
            best_ask = int(float(raw["best_ask"]) * 100)
            book.queue_sizes[price] = max(
                book.queue_sizes.get(price, 0) + size, 0
            )
            book.watchers[price].update_queue_pos(
                order_quantity=size,
                queue_size=book.queue_sizes[price],
                is_trade=False,
            )
            book.best_bid = best_bid
            book.best_ask = best_ask
            book.mid_price = (best_bid + best_ask) / 2.0
            self._update_microprice(book)
            self._update_mid_returns(book)
            self._update_trend(book)

        elif t == "last_trade":
            raw_side = raw.get("side")
            if raw_side == "BUY":
                side = 1
            elif raw_side == "SELL":
                side = 0
            else:
                side = int(raw_side)
            size = float(raw["size"])
            self._update_intensity(book.underlying, book.outcome, side, size)
            self._handle_trade(
                {"side": side, "size": size, "timestamp": ts}, book
            )

        out = self.alpha_function(msg)
        if out is not None:
            book.alpha_val = float(out)

    def _handle_trade(self, data, book: BookState):
        price = book.best_ask if data["side"] == 1 else book.best_bid
        if not (0 < price < 100):
            return

        watcher = book.watchers[price]
        if not watcher.orders:
            return

        filled = []
        remaining = float(data["size"])
        for order_id, qty in list(watcher.orders.items()):
            if watcher.queue_pos.get(order_id, float("inf")) <= 0 and remaining > 0:
                executed = min(float(qty), remaining)
                filled.append((order_id, executed))
                remaining -= executed

        for order_id, executed in filled:
            if data["side"] == 1:
                book.inventory -= executed
                self.wealth += executed * price
                our_side = 0
            else:
                book.inventory += executed
                self.wealth -= executed * price
                our_side = 1

            total_inv_value = sum(
                b.inventory * b.mid_price for b in self.books.values()
            )
            mtm = self.wealth - self._initial_wealth + total_inv_value
            print(
                f"[{book.underlying}/{book.outcome}] "
                f"{'BUY' if our_side == 1 else 'SELL'} {executed}@{price} | "
                f"inv={book.inventory:.0f} net={self.net_exposure(book.underlying):.0f} | "
                f"mtm={mtm:.1f}c | t={data['timestamp']}"
            )
            self.logger.log_msg({"type": "trader_state", "data": {
                "timestamp": data["timestamp"],
                "underlying": book.underlying,
                "outcome": book.outcome,
                "book_inventory": book.inventory,
                "net_exposure": self.net_exposure(book.underlying),
                "wealth": self.wealth,
                "mid_price": book.mid_price,
                "mtm_pnl": mtm,
            }})
            self.logger.log_msg({"type": "trade_log", "data": {
                "timestamp": data["timestamp"],
                "underlying": book.underlying,
                "outcome": book.outcome,
                "price": price,
                "side": our_side,
                "quantity": executed,
            }})
            watcher.remove_order(order_id)

        remaining_trade_size = float(data["size"]) - sum(ex for _, ex in filled)
        if watcher.orders and remaining_trade_size > 0:
            watcher.update_queue_pos(
                order_quantity=remaining_trade_size,
                queue_size=book.queue_sizes.get(price, 0),
                is_trade=True,
            )

    def _place_orders(
        self, book: BookState,
        reference_bid: int, reference_ask: int, scaled: float,
    ):
        soft_limit = self.inventory_limit - 1

        allow_bids = scaled < self.inventory_limit
        allow_asks = scaled > -self.inventory_limit
        if scaled >= soft_limit:
            allow_bids = False
        if scaled <= -soft_limit:
            allow_asks = False

        for i in range(0, self.level_range):
            bid_price = reference_bid - i
            ask_price = reference_ask + i

            if bid_price > 0 and allow_bids:
                if not book.watchers[bid_price].orders:
                    entry_pos = (
                        book.queue_sizes.get(bid_price, 0) * self.queue_entry_frac
                    )
                    order_id = self.sender.send((bid_price, 1, 1, book.underlying, book.outcome))
                    book.watchers[bid_price].add_order(order_id, 1, entry_pos)

            if ask_price < 100 and allow_asks:
                if not book.watchers[ask_price].orders:
                    entry_pos = (
                        book.queue_sizes.get(ask_price, 0) * self.queue_entry_frac
                    )
                    order_id = self.sender.send((ask_price, 1, 0, book.underlying, book.outcome))
                    book.watchers[ask_price].add_order(order_id, 1, entry_pos)

    def _cancel_stale_orders(
        self, book: BookState,
        reference_bid: int, reference_ask: int, scaled: float,
    ):
        soft_limit = self.inventory_limit - 1
        tolerance = self.level_range * 2

        for p in range(1, 100):
            if not book.watchers[p].orders:
                continue

            should_cancel = False

            if book.best_bid < p < book.best_ask:
                should_cancel = True

            if p > reference_ask + tolerance or p < reference_bid - tolerance:
                should_cancel = True

            if scaled >= soft_limit and p <= book.best_bid:
                should_cancel = True
            if scaled <= -soft_limit and p >= book.best_ask:
                should_cancel = True

            if should_cancel:
                for oid in list(book.watchers[p].orders):
                    if self.on_cancel:
                        self.on_cancel(oid)
                    book.watchers[p].remove_order(oid)

    def print_watchers(self):
        for (underlying, outcome), book in self.books.items():
            counts = [len(book.watchers[p].orders) for p in range(1, 100)]
            print(f"{underlying}/{outcome}: {counts}")


# ---------------------------------------------------------------------------
# Queue position tracker (one per price level, per book)
# ---------------------------------------------------------------------------

class QueueWatcher:
    def __init__(self, price_level: int, queue_pos_func: Callable):
        self.price_level = price_level
        self.orders: dict = {}
        self.queue_pos: dict = {}
        self.lambda_func = queue_pos_func

    def add_order(self, order_id: int, quantity: int, prior_queue_size: float):
        self.orders[order_id] = quantity
        self.queue_pos[order_id] = prior_queue_size

    def remove_order(self, order_id: int):
        self.orders.pop(order_id, None)
        self.queue_pos.pop(order_id, None)

    def partial_fill(self, order_id: int, new_quantity: int):
        self.orders[order_id] = new_quantity
        self.queue_pos[order_id] = 0

    def update_queue_pos(self, order_quantity: float, queue_size: float, is_trade: bool):
        if is_trade:
            for order_id in self.orders:
                self.queue_pos[order_id] = max(
                    self.queue_pos[order_id] - order_quantity, 0
                )
        else:
            for order_id in self.orders:
                V_old = self.queue_pos[order_id]
                if V_old == 0:
                    continue
                S = self.orders[order_id]
                V_new = max(
                    V_old - self._p(V_old, queue_size, S) * order_quantity, 0
                )
                self.queue_pos[order_id] = V_new

    def _p(self, V_old: float, queue_size: float, S: float) -> float:
        denom = self.lambda_func(V_old + max(queue_size - S - V_old, 0))
        if denom == 0:
            return 0.0
        return self.lambda_func(V_old) / denom


# ---------------------------------------------------------------------------
# Backtest helpers
# ---------------------------------------------------------------------------

class BacktestOID(TradeMsgSender):
    def __init__(self):
        self._oid = -1

    def send(self, msg) -> int:
        self._oid += 1
        return self._oid


class BacktestLogger(Logger):
    def __init__(self):
        self.trader_states: dict = defaultdict(list)
        self.trades: dict = defaultdict(list)
        self.settlements: list = []

    def log_msg(self, msg):
        if msg["type"] == "trader_state":
            d = msg["data"]
            self.trader_states["timestamp"].append(d["timestamp"])
            self.trader_states["underlying"].append(d["underlying"])
            self.trader_states["outcome"].append(d["outcome"])
            self.trader_states["book_inventory"].append(d["book_inventory"])
            self.trader_states["net_exposure"].append(d["net_exposure"])
            self.trader_states["wealth"].append(d["wealth"])
            self.trader_states["mid_price"].append(d["mid_price"])
            self.trader_states["mtm_pnl"].append(d["mtm_pnl"])
        elif msg["type"] == "trade_log":
            d = msg["data"]
            self.trades["timestamp"].append(d["timestamp"])
            self.trades["underlying"].append(d["underlying"])
            self.trades["outcome"].append(d["outcome"])
            self.trades["price"].append(d["price"])
            self.trades["side"].append(d["side"])
            self.trades["quantity"].append(d["quantity"])
        elif msg["type"] == "settlement":
            self.settlements.append(msg["data"])

    def df_parse(self):
        return (
            pd.DataFrame.from_dict(self.trader_states),
            pd.DataFrame.from_dict(self.trades),
        )


# ---------------------------------------------------------------------------
# Microprice regression coefficient computation
# ---------------------------------------------------------------------------

def compute_microprice_coefficients(
    book_df: pd.DataFrame,
    underlyings: list,
    lookback: int = 5,
) -> Dict[str, Tuple[float, float]]:
    """
    Fit logistic regression: P(uptick) ~ sigmoid(intercept + slope * microprice_dev)
    for each underlying using Up book snapshots.

    Returns {underlying: (slope, intercept)}.
    """
    from sklearn.linear_model import LogisticRegression

    px_array = np.arange(0, 101)
    coefficients = {}

    for underlying in underlyings:
        sub = book_df[
            (book_df["underlying"] == underlying) & (book_df["outcome"] == "Up")
        ][["timestamp", "sizes"]].sort_values("timestamp").reset_index(drop=True)

        if len(sub) < lookback + 2:
            print(f"  {underlying}: insufficient book data ({len(sub)} rows)")
            continue

        microprice_devs = []
        mids = []
        for i in range(len(sub)):
            sizes = np.array(sub.iloc[i]["sizes"])
            bid_px = px_array[sizes > 0]
            ask_px = px_array[sizes < 0]
            if len(bid_px) == 0 or len(ask_px) == 0:
                microprice_devs.append(np.nan)
                mids.append(np.nan)
                continue
            bb, ba = bid_px[-1], ask_px[0]
            bid_sz, ask_sz = float(sizes[bb]), float(abs(sizes[ba]))
            mid = (bb + ba) / 2.0
            microprice = (bid_sz * ba + ask_sz * bb) / (bid_sz + ask_sz)
            microprice_devs.append(microprice - mid)
            mids.append(mid)

        df_sig = pd.DataFrame({"microprice_dev": microprice_devs, "mid": mids})
        df_sig["microprice_dev"] = df_sig["microprice_dev"].rolling(lookback).mean()
        df_sig["mid_change"] = df_sig["mid"].diff().shift(-1)
        df_sig = df_sig.dropna()
        df_sig = df_sig[df_sig["mid_change"] != 0]

        if len(df_sig) < 20:
            print(f"  {underlying}: insufficient non-zero ticks ({len(df_sig)})")
            continue

        X = df_sig[["microprice_dev"]].values
        y = (df_sig["mid_change"] > 0).astype(int).values

        lr = LogisticRegression(solver="lbfgs")
        lr.fit(X, y)
        slope = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])
        print(f"  {underlying}: slope={slope:.4f} intercept={intercept:.4f} "
              f"acc={lr.score(X, y):.4f} n={len(df_sig)}")
        coefficients[underlying] = (slope, intercept)

    return coefficients


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def backtest(
    dataframes_by_book: Dict[Tuple[str, str], dict],
    model: TradingModel,
    settlement_prices: Optional[Dict[str, float]] = None,
    **controller_kwargs,
):
    """
    Run a full backtest over merged message streams from multiple
    (underlying, outcome) books.

    Parameters
    ----------
    dataframes_by_book : {("btc","Up"): {"book":df, "price_change":df, "last_trade":df}, ...}
    model              : TradingModel instance
    settlement_prices  : {underlying: up_settlement_price}, None = infer
    controller_kwargs  : Forwarded to Controller

    Returns
    -------
    (final_wealth, controller, logger)
    """
    def merged_stream(dfs_by_book):
        def gen(df, src):
            for row in df.itertuples(index=False):
                d = row._asdict()
                yield d["timestamp"], {
                    "timestamp": d["timestamp"],
                    "type": src,
                    "data": dict(d),
                }
        streams = []
        for key, dfs in dfs_by_book.items():
            for src, df in dfs.items():
                streams.append(gen(df, src))
        for _, msg in heapq.merge(*streams, key=lambda x: x[0]):
            yield msg

    logger = BacktestLogger()
    controller = Controller(
        model.run,
        BacktestOID(),
        logger,
        live=False,
        **controller_kwargs,
    )
    for underlying, outcome in dataframes_by_book:
        controller.add_book(underlying, outcome)

    for msg in merged_stream(dataframes_by_book):
        controller.parse(msg)

    # Settle
    if settlement_prices is None:
        settlement_prices = {}
        for underlying in controller._underlyings:
            up_book = controller.books.get((underlying, "Up"))
            if up_book:
                settlement_prices[underlying] = (
                    100.0 if up_book.best_bid >= 50 else 0.0
                )
            else:
                settlement_prices[underlying] = 50.0
    controller.settle(settlement_prices)

    return controller.wealth, controller, logger


def generate_last_trade(
    last_trade_df: pd.DataFrame,
    window_size: int = 15 * 60 * 1000,
    p: float = 0.5,
):
    """
    Synthesise market order arrivals via thinned Poisson process with KDE intensity.

    NOTE: Caller must add "outcome" and "underlying" columns to the returned
    DataFrame before passing into the backtest.
    """
    time_floor = (last_trade_df["timestamp"].min() // window_size) * window_size

    all_offsets = last_trade_df["timestamp"].values.astype(float) - time_floor
    offset_min = float(all_offsets.min())
    offset_max = float(all_offsets.max())

    buys = (
        last_trade_df[last_trade_df["side"] == "BUY"]["timestamp"].values.astype(float)
        - time_floor
    )
    sells = (
        last_trade_df[last_trade_df["side"] == "SELL"]["timestamp"].values.astype(float)
        - time_floor
    )

    _empty = pd.DataFrame(columns=["timestamp", "side", "size", "against_user"])
    if len(buys) < 2 or len(sells) < 2:
        return _empty

    window = np.linspace(offset_min, offset_max, max(len(last_trade_df), 100))
    kde_buy = gaussian_kde(buys)
    kde_sell = gaussian_kde(sells)

    intensity_buy = kde_buy(window) * len(buys) * (offset_max - offset_min)
    intensity_sell = kde_sell(window) * len(sells) * (offset_max - offset_min)
    buy_sup = intensity_buy.max()
    sell_sup = intensity_sell.max()

    log_sizes = np.log(np.abs(last_trade_df["size"].values).clip(1e-6))
    kde_size = gaussian_kde(log_sizes)

    n_buys = poisson.rvs(buy_sup)
    n_sells = poisson.rvs(sell_sup)

    buy_times = np.random.uniform(offset_min, offset_max, size=n_buys)
    sell_times = np.random.uniform(offset_min, offset_max, size=n_sells)

    buy_accept = np.interp(buy_times, window, intensity_buy / buy_sup)
    sell_accept = np.interp(sell_times, window, intensity_sell / sell_sup)

    buy_times = buy_times[np.random.uniform(size=n_buys) <= buy_accept]
    sell_times = sell_times[np.random.uniform(size=n_sells) <= sell_accept]

    def make_side_df(times, side_val):
        if len(times) == 0:
            return _empty.copy()
        sizes = np.exp(kde_size.resample(len(times)).squeeze(0)).astype(int).clip(1)
        return pd.DataFrame({
            "timestamp": times,
            "side": np.full(len(times), float(side_val)),
            "size": sizes,
            "against_user": np.random.uniform(size=len(times)) <= p,
        })

    df = pd.concat(
        [make_side_df(buy_times, 1), make_side_df(sell_times, 0)], ignore_index=True
    )
    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Entry point — multi-asset backtest with signals
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from python.utils.gdrive_utils import GDriveUploader
    from python.utils.consts_utils import GDRIVE_FOLDER_ID
    from python.utils.parser_utils import parse_df
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    ASSETS = ["btc", "eth"]
    WINDOW = 15 * 60 * 1000  # ms

    # --- Load data ---
    gdrive = GDriveUploader()
    files = gdrive.list_files_in_folder(GDRIVE_FOLDER_ID)
    dfs_by_type = defaultdict(list)

    for f in files[:1]:
        print(f"Processing {f['name']} ...")
        dfs = gdrive.read_parquets_from_tar_zst(f["id"])
        try:
            for df in dfs:
                file_type = df["__source_file"].iloc[0].rsplit("/")[1].split(".")[0]
                df = parse_df(df=df.drop(columns="__source_file"), file_type=file_type)
                dfs_by_type[file_type].append(df)
        except Exception as e:
            print(f"  skipped: {e}")
            continue

    # --- Build per-(underlying, outcome) dataframes ---
    def normalise_time(df):
        df = df.copy()
        floor = (df["timestamp"].min() // WINDOW) * WINDOW
        df["timestamp"] -= floor
        return df

    def filt(df, underlying, outcome):
        return df[(df["underlying"] == underlying) & (df["outcome"] == outcome)]

    class NaiveModel(TradingModel):
        def run(self, msg):
            return 0.0

    dataframes_by_book = {}
    for underlying in ASSETS:
        for outcome in ["Up", "Down"]:
            lt_hist = filt(dfs_by_type["last_trade"][0], underlying, outcome)
            if lt_hist.empty or len(lt_hist) < 4:
                continue
            pc_raw = filt(dfs_by_type["price_change"][0], underlying, outcome)
            bk_raw = filt(dfs_by_type["book"][0], underlying, outcome)
            if pc_raw.empty or bk_raw.empty:
                continue

            lt_synth = generate_last_trade(lt_hist)
            lt_synth["outcome"] = outcome
            lt_synth["underlying"] = underlying

            dataframes_by_book[(underlying, outcome)] = {
                "book": normalise_time(bk_raw),
                "price_change": normalise_time(pc_raw),
                "last_trade": lt_synth,
            }

    active_assets = sorted(set(u for u, _ in dataframes_by_book.keys()))
    print(f"\nActive assets: {active_assets}")
    print(f"Active books: {list(dataframes_by_book.keys())}")

    if not dataframes_by_book:
        print("No data for any book.")
    else:
        # --- Run backtest ---
        final_wealth, ctrl, logger = backtest(
            dataframes_by_book,
            NaiveModel(),
            inventory_limit=10,
            level_range=3,
            skew_per_unit=1.0,
            window_ms=WINDOW,
            queue_entry_frac=0.15,
            intensity_weight=0.5,
        )

        state_df, trades_df = logger.df_parse()

        # --- Summary ---
        for sett in logger.settlements:
            print(
                f"  {sett['underlying']}/{sett['outcome']}: "
                f"settle@{sett['settlement_price']}  "
                f"inv={sett['inventory_settled']:.0f}  "
                f"pnl={sett['pnl_from_inventory']:.1f}c"
            )
        for u in active_assets:
            print(f"  {u} net_exposure: {ctrl.net_exposure(u):.0f}")
        print(f"\nFinal: wealth={final_wealth:.1f}c  n_trades={len(trades_df)}")

        # --- Plot ---
        # Per asset: net exposure, Up price+trades, Down price+trades (3 rows).
        # Shared: combined MTM PnL (1 row).
        n_rows = len(active_assets) * 3 + 1
        fig = plt.figure(figsize=(14, 3.5 * n_rows))
        gs = gridspec.GridSpec(n_rows, 1, hspace=0.55)

        row = 0
        for underlying in active_assets:
            # --- net exposure ---
            ax = fig.add_subplot(gs[row])
            asset_states = (
                state_df[state_df["underlying"] == underlying]
                if not state_df.empty else state_df
            )
            if not asset_states.empty:
                ax.plot(asset_states["timestamp"], asset_states["net_exposure"], color="blue")
            ax.axhline(0, color="grey", lw=0.8, ls="--")
            n_t = len(trades_df[trades_df["underlying"] == underlying]) if not trades_df.empty else 0
            ax.set_title(f"{underlying.upper()} Net Exposure  ({n_t} trades)")
            ax.set_ylabel("net exp")
            row += 1

            # --- price + trades for each outcome ---
            for outcome in ["Up", "Down"]:
                key = (underlying, outcome)
                if key not in dataframes_by_book:
                    continue
                ax = fig.add_subplot(gs[row])
                pc_df = dataframes_by_book[key]["price_change"]
                ax.plot(pc_df["timestamp"], pc_df["best_ask"], color="green", alpha=0.5, label="ask")
                ax.plot(pc_df["timestamp"], pc_df["best_bid"], color="red", alpha=0.5, label="bid")
                oc_trades = (
                    trades_df[
                        (trades_df["underlying"] == underlying)
                        & (trades_df["outcome"] == outcome)
                    ]
                    if not trades_df.empty else trades_df
                )
                if not oc_trades.empty:
                    buys_t = oc_trades[oc_trades["side"] == 1]
                    sells_t = oc_trades[oc_trades["side"] == 0]
                    ax.scatter(
                        buys_t["timestamp"], buys_t["price"],
                        marker="^", color="green", s=50, zorder=5, label="our buys",
                    )
                    ax.scatter(
                        sells_t["timestamp"], sells_t["price"],
                        marker="v", color="red", s=50, zorder=5, label="our sells",
                    )
                ax.set_title(f"{underlying.upper()} {outcome} — price + our trades")
                ax.legend(fontsize=7)
                row += 1

        # Combined MTM PnL
        ax = fig.add_subplot(gs[row])
        if not state_df.empty:
            ax.plot(state_df["timestamp"], state_df["mtm_pnl"], color="purple")
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_title("Combined Mark-to-Market PnL (all assets)")
        ax.set_ylabel("PnL (cents)")

        plt.suptitle(
            "Multi-Asset Market Maker — Intensity + Correlation",
            y=1.01, fontsize=13,
        )
        plt.tight_layout()
        plt.show()
