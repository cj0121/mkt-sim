from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import json
import time

import requests
import numpy as np


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

    def resolve_S0(self, S0: Optional[float] = None, symbol: str = "SPY") -> Optional[float]:
        if S0 is not None and np.isfinite(S0):
            return float(S0)
        return self.fetch_price(symbol)


__all__ = ["AlphaVantageClient"]


