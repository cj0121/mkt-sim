from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


def _ensure_datetime64ns_to_int64_nanos(values: Iterable[Any]) -> NDArray[np.int64]:
    """Convert various date-like inputs into int64 epoch nanoseconds.

    Accepted inputs:
    - numpy datetime64 array (any unit) → coerced to ns, then viewed as int64
    - sequence of ISO-like strings → parsed by numpy to datetime64[ns], then viewed as int64
    - int-like array → assumed to already be epoch nanoseconds (no unit auto-detect)

    This function purposefully avoids guessing units for integers. If integers are
    not already in nanoseconds, convert them before calling this function.
    """
    arr = np.asarray(values)

    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]").view("int64")

    if np.issubdtype(arr.dtype, np.integer):
        return np.ascontiguousarray(arr, dtype=np.int64)

    # Fallback: let numpy parse strings/objects to datetime64[ns]
    return np.asarray(values, dtype="datetime64[ns]").view("int64")


def _ensure_float64(values: Optional[Iterable[Any]]) -> Optional[NDArray[np.float64]]:
    if values is None:
        return None
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64), dtype=np.float64)


def _validate_parallel_lengths(length: int, arrays: Sequence[Optional[np.ndarray]], names: Sequence[str]) -> None:
    for a, name in zip(arrays, names):
        if a is None:
            continue
        if a.shape[0] != length:
            raise ValueError(f"Array '{name}' length {a.shape[0]} != expected {length}")


@dataclass(slots=True, frozen=True)
class StockSeries:
    """Thin, immutable container for a single symbol's time series.

    Canonical in-memory schema:
    - ts_ns: int64 epoch nanoseconds, strictly increasing and unique
    - close: float64 prices
    Optional parallel arrays are float64 and match length when provided.
    """

    symbol: str
    ts_ns: NDArray[np.int64]
    close: NDArray[np.float64]
    open: Optional[NDArray[np.float64]] = None
    high: Optional[NDArray[np.float64]] = None
    low: Optional[NDArray[np.float64]] = None
    adj_close: Optional[NDArray[np.float64]] = None
    volume: Optional[NDArray[np.float64]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # manual cache slots for derived arrays (compatible with slots + frozen)
    _log_close: Optional[NDArray[np.float64]] = field(default=None, init=False, repr=False, compare=False)
    _log_returns: Optional[NDArray[np.float64]] = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Basic dtype checks
        if not (isinstance(self.ts_ns, np.ndarray) and self.ts_ns.dtype == np.int64):
            raise TypeError("ts_ns must be a numpy ndarray with dtype int64 (epoch ns)")
        if not (isinstance(self.close, np.ndarray) and self.close.dtype == np.float64):
            raise TypeError("close must be a numpy ndarray with dtype float64")

        length = self.close.shape[0]
        if self.ts_ns.shape[0] != length:
            raise ValueError("ts_ns and close must have the same length")

        _validate_parallel_lengths(
            length,
            arrays=[self.open, self.high, self.low, self.adj_close, self.volume],
            names=["open", "high", "low", "adj_close", "volume"],
        )

        if length == 0:
            raise ValueError("Series must be non-empty")

        # Monotonic strictly increasing timestamps
        diffs = np.diff(self.ts_ns)
        if not np.all(diffs > 0):
            raise ValueError("ts_ns must be strictly increasing and unique")

        # Quick NaN check for required arrays
        if np.isnan(self.close).any():
            raise ValueError("close contains NaN values; clean before constructing")

    # ---------- Constructors ----------
    @classmethod
    def from_arrays(
        cls,
        symbol: str,
        ts_ns: Iterable[Any],
        close: Iterable[Any],
        *,
        open: Optional[Iterable[Any]] = None,
        high: Optional[Iterable[Any]] = None,
        low: Optional[Iterable[Any]] = None,
        adj_close: Optional[Iterable[Any]] = None,
        volume: Optional[Iterable[Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "StockSeries":
        ts_ns_arr = _ensure_datetime64ns_to_int64_nanos(ts_ns)
        close_arr = _ensure_float64(close)
        open_arr = _ensure_float64(open)
        high_arr = _ensure_float64(high)
        low_arr = _ensure_float64(low)
        adj_arr = _ensure_float64(adj_close)
        vol_arr = _ensure_float64(volume)

        return cls(
            symbol=symbol,
            ts_ns=ts_ns_arr,
            close=close_arr,  # type: ignore[arg-type]
            open=open_arr,
            high=high_arr,
            low=low_arr,
            adj_close=adj_arr,
            volume=vol_arr,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def from_lists(
        cls,
        symbol: str,
        dates: Iterable[Any],
        close: Iterable[Any],
        **optionals: Any,
    ) -> "StockSeries":
        ts_ns = _ensure_datetime64ns_to_int64_nanos(dates)
        return cls.from_arrays(symbol, ts_ns=ts_ns, close=close, **optionals)

    @classmethod
    def from_pandas(
        cls,
        symbol: str,
        df: Any,
        *,
        date_col: str = "timestamp",
        close_col: str = "close",
        open_col: Optional[str] = "open",
        high_col: Optional[str] = "high",
        low_col: Optional[str] = "low",
        adj_close_col: Optional[str] = None,
        volume_col: Optional[str] = "volume",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "StockSeries":
        # Import locally to keep pandas out of the import path for users not using it
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pandas is required for from_pandas()") from exc

        if not hasattr(df, "__getitem__"):
            raise TypeError("df must be a pandas DataFrame-like with column access")

        ts_vals = df[date_col].to_numpy()
        ts_ns = _ensure_datetime64ns_to_int64_nanos(ts_vals)
        close_vals = df[close_col].to_numpy()

        def pick(col: Optional[str]) -> Optional[np.ndarray]:
            if col is None or col not in df.columns:
                return None
            return df[col].to_numpy()

        return cls.from_arrays(
            symbol=symbol,
            ts_ns=ts_ns,
            close=close_vals,
            open=pick(open_col),
            high=pick(high_col),
            low=pick(low_col),
            adj_close=pick(adj_close_col),
            volume=pick(volume_col),
            metadata=metadata,
        )

    @classmethod
    def from_records(
        cls,
        symbol: str,
        records: Iterable[Mapping[str, Any]],
        *,
        time_field: str,
        close_field: str = "close",
        open_field: Optional[str] = "open",
        high_field: Optional[str] = "high",
        low_field: Optional[str] = "low",
        adj_close_field: Optional[str] = None,
        volume_field: Optional[str] = "volume",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "StockSeries":
        recs = list(records)
        ts_vals = [r[time_field] for r in recs]
        ts_ns = _ensure_datetime64ns_to_int64_nanos(ts_vals)
        close_vals = [r[close_field] for r in recs]

        def opt(field: Optional[str]) -> Optional[Iterable[Any]]:
            if field is None:
                return None
            return [r[field] for r in recs]

        return cls.from_arrays(
            symbol=symbol,
            ts_ns=ts_ns,
            close=close_vals,
            open=opt(open_field),
            high=opt(high_field),
            low=opt(low_field),
            adj_close=opt(adj_close_field),
            volume=opt(volume_field),
            metadata=metadata,
        )

    # ---------- Provider adapters ----------
    @classmethod
    def from_alphavantage(
        cls,
        symbol: str,
        payload: Any,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "StockSeries":
        # Handle tuple/list payload shaped like (dates, closes)
        if isinstance(payload, (list, tuple)) and len(payload) == 2:
            dates, closes = payload
            ts_ns = _ensure_datetime64ns_to_int64_nanos(dates)
            close_vals = np.asarray(closes, dtype=np.float64)
            # Ensure ascending order by time if not already
            order = np.argsort(ts_ns)
            if not np.all(order == np.arange(ts_ns.shape[0])):
                ts_ns = ts_ns[order]
                close_vals = close_vals[order]
            return cls.from_arrays(symbol, ts_ns=ts_ns, close=close_vals, metadata=metadata)

        # Handle pandas DataFrame with typical columns
        if hasattr(payload, "__getitem__") and hasattr(payload, "columns"):
            cols = set(getattr(payload, "columns", []))
            date_col = "timestamp" if "timestamp" in cols else ("date" if "date" in cols else next(iter(cols)))
            close_col = "close" if "close" in cols else ("adjusted_close" if "adjusted_close" in cols else None)
            if close_col is None:
                raise ValueError("AlphaVantage payload must include a 'close' or 'adjusted_close' column")
            return cls.from_pandas(symbol, payload, date_col=date_col, close_col=close_col, metadata=metadata)

        # Handle dict payload from AlphaVantage Time Series endpoints
        if isinstance(payload, dict):
            series_key = None
            for key in payload.keys():
                if "Time Series" in key:
                    series_key = key
                    break
            if series_key is None:
                raise ValueError("Unrecognized AlphaVantage payload: missing 'Time Series' key")

            ts_dict = payload[series_key]
            dates = []
            closes = []
            opens = []
            highs = []
            lows = []
            vols = []
            for k, v in ts_dict.items():
                dates.append(k)
                # AlphaVantage uses numeric-string keys like '1. open'
                opens.append(float(v.get("1. open", v.get("open", np.nan))))
                highs.append(float(v.get("2. high", v.get("high", np.nan))))
                lows.append(float(v.get("3. low", v.get("low", np.nan))))
                closes.append(float(v.get("4. close", v.get("close", v.get("5. adjusted close", np.nan)))))
                vols.append(float(v.get("6. volume", v.get("volume", np.nan))))

            order = np.argsort(np.asarray(dates, dtype="datetime64[ns]").view("int64"))
            dates = [dates[i] for i in order]
            opens = [opens[i] for i in order]
            highs = [highs[i] for i in order]
            lows = [lows[i] for i in order]
            closes = [closes[i] for i in order]
            vols = [vols[i] for i in order]

            return cls.from_lists(
                symbol,
                dates=dates,
                close=closes,
                open=opens,
                high=highs,
                low=lows,
                volume=vols,
                metadata=metadata,
            )

        raise TypeError("Unsupported AlphaVantage payload type; expected DataFrame or dict")

    @classmethod
    def from_polygon(
        cls,
        symbol: str,
        payload: Any,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "StockSeries":
        # DataFrame path
        if hasattr(payload, "__getitem__") and hasattr(payload, "columns"):
            cols = set(getattr(payload, "columns", []))
            # Polygon aggregates use 't' for timestamp (ms) and 'c' for close
            if "t" in cols and "c" in cols:
                df = payload.copy()
                t_vals = np.asarray(df["t"], dtype=np.int64) * 1_000_000
                df = df.rename(columns={"c": "close"})
                df["timestamp"] = t_vals.view("datetime64[ns]")
                return cls.from_pandas(symbol, df, date_col="timestamp", close_col="close", metadata=metadata)
            # Fallback guess
            date_col = "timestamp" if "timestamp" in cols else ("date" if "date" in cols else next(iter(cols)))
            close_col = "close" if "close" in cols else ("c" if "c" in cols else None)
            if close_col is None:
                raise ValueError("Polygon payload must include a close price column")
            return cls.from_pandas(symbol, payload, date_col=date_col, close_col=close_col, metadata=metadata)

        # JSON dict path: typical /v2/aggs results
        if isinstance(payload, dict) and "results" in payload:
            recs = payload.get("results", [])
            if not recs:
                raise ValueError("Polygon payload has no 'results'")
            dates = [int(r["t"]) * 1_000_000 for r in recs]  # ms → ns
            closes = [float(r["c"]) for r in recs]
            opens = [float(r.get("o", np.nan)) for r in recs]
            highs = [float(r.get("h", np.nan)) for r in recs]
            lows = [float(r.get("l", np.nan)) for r in recs]
            vols = [float(r.get("v", np.nan)) for r in recs]

            order = np.argsort(np.asarray(dates, dtype=np.int64))
            dates = [dates[i] for i in order]
            opens = [opens[i] for i in order]
            highs = [highs[i] for i in order]
            lows = [lows[i] for i in order]
            closes = [closes[i] for i in order]
            vols = [vols[i] for i in order]

            return cls.from_arrays(
                symbol,
                ts_ns=np.asarray(dates, dtype=np.int64),
                close=closes,
                open=opens,
                high=highs,
                low=lows,
                volume=vols,
                metadata=metadata,
            )

        raise TypeError("Unsupported Polygon payload type; expected DataFrame or dict with 'results'")

    # ---------- Accessors & utilities ----------
    @property
    def n(self) -> int:
        return int(self.close.shape[0])

    def as_arrays(self) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
        return self.ts_ns, self.close

    def window(self, start: int, end: int) -> "StockSeries":
        return StockSeries(
            symbol=self.symbol,
            ts_ns=self.ts_ns[start:end],
            close=self.close[start:end],
            open=None if self.open is None else self.open[start:end],
            high=None if self.high is None else self.high[start:end],
            low=None if self.low is None else self.low[start:end],
            adj_close=None if self.adj_close is None else self.adj_close[start:end],
            volume=None if self.volume is None else self.volume[start:end],
            metadata=self.metadata,
        )

    def slice_by_time(self, start_ns: Optional[int], end_ns: Optional[int]) -> "StockSeries":
        ts = self.ts_ns
        if start_ns is None:
            start_idx = 0
        else:
            start_idx = int(np.searchsorted(ts, start_ns, side="left"))
        if end_ns is None:
            end_idx = ts.shape[0]
        else:
            end_idx = int(np.searchsorted(ts, end_ns, side="left"))
        return self.window(start_idx, end_idx)

    def slice_by_date(self, start: Optional[Any] = None, end: Optional[Any] = None) -> "StockSeries":
        """Slice by date-like bounds (e.g., "YYYY-MM-DD").

        Semantics match slice_by_time: start-inclusive, end-exclusive using left bounds.
        Accepts strings, numpy.datetime64, or datetime.date/datetime.
        """
        def to_ns(val: Optional[Any]) -> Optional[int]:
            if val is None:
                return None
            try:
                v = np.datetime64(val, "ns")
            except Exception:
                v = np.datetime64(str(val), "ns")
            return int(v.view("int64"))

        return self.slice_by_time(to_ns(start), to_ns(end))

    def as_datetime64(self) -> NDArray[np.datetime64]:
        return self.ts_ns.view("datetime64[ns]")

    def as_date_strings(self, fmt: Optional[str] = "%Y-%m-%d") -> Sequence[str]:
        if fmt in (None, "%Y-%m-%d"):
            dt = self.as_datetime64().astype("datetime64[D]")
            return [str(d) for d in dt]
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Custom fmt requires pandas; install pandas or use default format") from exc
        return list(pd.to_datetime(self.as_datetime64()).strftime(fmt))

    # ---------- Cached derived arrays ----------
    @property
    def log_close(self) -> NDArray[np.float64]:
        val = self._log_close
        if val is None:
            val = np.log(self.close)
            object.__setattr__(self, "_log_close", val)
        return val

    @property
    def log_returns(self) -> NDArray[np.float64]:
        val = self._log_returns
        if val is None:
            val = np.diff(self.log_close)
            object.__setattr__(self, "_log_returns", val)
        return val


__all__ = ["StockSeries"]


