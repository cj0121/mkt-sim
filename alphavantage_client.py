from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import json
import time

import requests
import numpy as np
import warnings


ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"


def _to_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
        if np.isfinite(f):
            return f
        return None
    except Exception:
        return None


@dataclass
class AlphaVantageClient:
    api_key: Optional[str] = None
    timeout_sec: float = 10.0
    max_retries: int = 1

    def _get_key(self) -> Optional[str]:
        return self.api_key or os.getenv("ALPHAVANTAGE_API_KEY")

    def _request(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = self._get_key()
        if key is None:
            return None
        p = dict(params)
        p["apikey"] = key

        last_exc: Optional[Exception] = None
        for _ in range(max(1, self.max_retries)):
            try:
                resp = requests.get(ALPHAVANTAGE_BASE_URL, params=p, timeout=self.timeout_sec)
                if resp.status_code != 200:
                    last_exc = RuntimeError(f"HTTP {resp.status_code}")
                    time.sleep(0.25)
                    continue
                data = resp.json()
                # API rate limit or error messages come as plain keys
                if isinstance(data, dict) and any(k in data for k in ("Error Message", "Note", "Information")):
                    last_exc = RuntimeError(json.dumps(data))
                    time.sleep(0.25)
                    continue
                return data if isinstance(data, dict) else None
            except Exception as e:
                last_exc = e
                time.sleep(0.25)
        return None

    def fetch_global_quote(self, symbol: str) -> Optional[float]:
        data = self._request({"function": "GLOBAL_QUOTE", "symbol": symbol})
        if not data:
            return None
        quote = data.get("Global Quote") or data.get("globalQuote")
        if not isinstance(quote, dict):
            return None
        # Keys may be formatted as "05. price" or variations
        for key in ("05. price", "price", "l", "c"):
            if key in quote:
                val = _to_float(quote[key])
                if val is not None:
                    return val
        return None

    def fetch_intraday_latest(self, symbol: str, interval: str = "1min") -> Optional[float]:
        data = self._request({
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": "compact",
        })
        if not data:
            return None
        key = f"Time Series ({interval})"
        series = data.get(key)
        if not isinstance(series, dict) or not series:
            return None
        # The latest timestamp should be the first when sorted desc
        latest_ts = sorted(series.keys())[-1]
        bar = series.get(latest_ts, {})
        for k in ("4. close", "5. adjusted close", "close"):
            if k in bar:
                val = _to_float(bar[k])
                if val is not None:
                    return val
        return None

    def fetch_daily_latest(self, symbol: str) -> Optional[float]:
        data = self._request({
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "compact",
        })
        if not data:
            return None
        series = data.get("Time Series (Daily)")
        if not isinstance(series, dict) or not series:
            return None
        latest_date = sorted(series.keys())[-1]
        bar = series.get(latest_date, {})
        for k in ("5. adjusted close", "4. close", "close"):
            if k in bar:
                val = _to_float(bar[k])
                if val is not None:
                    return val
        return None

    def fetch_price(self, symbol: str) -> Optional[float]:
        # Prefer real-time quote, then intraday, then daily adjusted
        for fn in (self.fetch_global_quote, self.fetch_intraday_latest, self.fetch_daily_latest):
            try:
                val = fn(symbol)
                if val is not None:
                    return val
            except Exception:
                continue
        return None

    def fetch_daily_series(self, symbol: str, *, adjusted: bool = True, outputsize: str = "compact") -> Optional[tuple]:
        """Fetch daily closing price series for a symbol.

        Returns (dates, closes) where dates is np.ndarray[datetime64[D]] and closes is np.ndarray[float],
        both sorted ascending by date. If adjusted=True, uses TIME_SERIES_DAILY_ADJUSTED and prefers
        adjusted close; otherwise uses TIME_SERIES_DAILY.
        """
        def _parse_daily(data_obj: Optional[Dict[str, Any]], prefer_adjusted: bool) -> Optional[tuple]:
            if not data_obj:
                return None
            series = data_obj.get("Time Series (Daily)")
            if not isinstance(series, dict) or not series:
                return None
            items = sorted(series.items())
            dates = np.array([np.datetime64(k, "D") for k, _ in items])
            if prefer_adjusted:
                closes = np.array([
                    _to_float(v.get("5. adjusted close")) or _to_float(v.get("4. close")) or np.nan
                    for _, v in items
                ], dtype=float)
            else:
                closes = np.array([
                    _to_float(v.get("4. close")) or np.nan
                    for _, v in items
                ], dtype=float)
            mask = np.isfinite(closes)
            if not np.any(mask):
                return None
            return dates[mask], closes[mask]

        # Try adjusted series first if requested; fallback to unadjusted with a warning
        if adjusted:
            data_adj = self._request({
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "outputsize": outputsize,
            })
            parsed = _parse_daily(data_adj, prefer_adjusted=True)
            if parsed is not None:
                return parsed
            # Fallback to unadjusted
            warnings.warn("AlphaVantage adjusted series unavailable; falling back to unadjusted TIME_SERIES_DAILY.")
            data_unadj = self._request({
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": outputsize,
            })
            return _parse_daily(data_unadj, prefer_adjusted=False)
        else:
            data_unadj = self._request({
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": outputsize,
            })
            return _parse_daily(data_unadj, prefer_adjusted=False)

        # Drop NaNs if any
        mask = np.isfinite(closes)
        if not np.any(mask):
            return None
        return dates[mask], closes[mask]

    def resolve_S0(self, S0: Optional[float] = None, symbol: str = "SPY") -> Optional[float]:
        if S0 is not None and np.isfinite(S0):
            return float(S0)
        return self.fetch_price(symbol)


def fetch_price(symbol: str, api_key: Optional[str] = None) -> Optional[float]:
    return AlphaVantageClient(api_key=api_key).fetch_price(symbol)


def fetch_daily_series(symbol: str, *, adjusted: bool = True, outputsize: str = "compact", api_key: Optional[str] = None) -> Optional[tuple]:
    return AlphaVantageClient(api_key=api_key).fetch_daily_series(symbol, adjusted=adjusted, outputsize=outputsize)


def resolve_S0(S0: Optional[float] = None, symbol: str = "SPY", api_key: Optional[str] = None) -> Optional[float]:
    return AlphaVantageClient(api_key=api_key).resolve_S0(S0=S0, symbol=symbol)


__all__ = [
    "AlphaVantageClient",
    "fetch_price",
    "fetch_daily_series",
    "resolve_S0",
]


