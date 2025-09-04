from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os
import numpy as np


try:
    from polygon import RESTClient  # type: ignore
    _HAS_POLYGON = True
except Exception:
    _HAS_POLYGON = False


def _extract_snapshot_price(snap_obj) -> Optional[float]:
    lt = getattr(snap_obj, "lastTrade", getattr(snap_obj, "last_trade", None))
    if lt is not None:
        price = getattr(lt, "price", getattr(lt, "p", None))
        if price is not None and np.isfinite(price):
            return float(price)
    m = getattr(snap_obj, "min", getattr(snap_obj, "minute", None))
    if m is not None:
        cval = getattr(m, "c", getattr(m, "close", None))
        if cval is not None and np.isfinite(cval):
            return float(cval)
    d = getattr(snap_obj, "day", None)
    if d is not None:
        cval = getattr(d, "c", getattr(d, "close", None))
        if cval is not None and np.isfinite(cval):
            return float(cval)
    pd = getattr(snap_obj, "prevDay", getattr(snap_obj, "prev_day", None))
    if pd is not None:
        cval = getattr(pd, "c", getattr(pd, "close", None))
        if cval is not None and np.isfinite(cval):
            return float(cval)
    val = getattr(snap_obj, "value", None)
    if val is not None and np.isfinite(val):
        return float(val)
    return None


@dataclass
class PolygonClient:
    api_key: Optional[str] = None

    def _client(self):
        if not _HAS_POLYGON:
            return None
        key = self.api_key or os.getenv("POLYGON_API_KEY")
        if key is None:
            return None
        try:
            return RESTClient(api_key=key)  # type: ignore[name-defined]
        except Exception:
            return None

    def fetch_price(self, ticker: str) -> Optional[float]:
        c = self._client()
        if c is None:
            return None
        # 1) Single-ticker snapshot for stocks; snapshot v2 fallback
        try:
            if not ticker.startswith("I:"):
                get_snap_ticker = getattr(c, "get_snapshot_ticker", None)
                if callable(get_snap_ticker):
                    s = get_snap_ticker(ticker)
                    price = _extract_snapshot_price(s)
                    if price is not None:
                        return price
            snap_v2 = getattr(c, "get_snapshot_v2", None)
            if callable(snap_v2):
                category = "indices" if ticker.startswith("I:") else "stocks"
                s2 = snap_v2(category, ticker)
                price = _extract_snapshot_price(s2)
                if price is not None:
                    return price
        except Exception:
            pass
        # 2) Previous close
        try:
            prev = c.get_previous_close(ticker, adjusted=True)
            results = getattr(prev, "results", None)
            if results:
                last = results[-1]
                cval = getattr(last, "c", None)
                if cval is not None and np.isfinite(cval):
                    return float(cval)
        except Exception:
            pass
        # 3) Today's minute aggregates
        try:
            from datetime import datetime, timezone
            today = datetime.now(timezone.utc).date().isoformat()
            aggs = c.get_aggregates(ticker, 1, "minute", today, today, limit=50_000)
            results = getattr(aggs, "results", None)
            if results:
                cval = getattr(results[-1], "c", None)
                if cval is not None and np.isfinite(cval):
                    return float(cval)
        except Exception:
            pass
        return None

    def fetch_spx(self) -> Optional[float]:
        return self.fetch_price("I:SPX")

    def resolve_S0(self, S0: Optional[float] = None, ticker: str = "I:SPX") -> Optional[float]:
        if S0 is not None and np.isfinite(S0):
            return float(S0)
        return self.fetch_price(ticker)


# Functional helpers for convenience
def fetch_price_polygon(ticker: str, api_key: Optional[str] = None) -> Optional[float]:
    return PolygonClient(api_key=api_key).fetch_price(ticker)


def fetch_spx_price_polygon(api_key: Optional[str] = None) -> Optional[float]:
    return PolygonClient(api_key=api_key).fetch_spx()


def resolve_S0(S0: Optional[float] = None, api_key: Optional[str] = None, ticker: str = "I:SPX") -> Optional[float]:
    return PolygonClient(api_key=api_key).resolve_S0(S0=S0, ticker=ticker)


__all__ = [
    "PolygonClient",
    "fetch_price_polygon",
    "fetch_spx_price_polygon",
    "resolve_S0",
]


