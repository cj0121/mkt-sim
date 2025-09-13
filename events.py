from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _to_datetime64D(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[D]")
    return np.asarray(values, dtype="datetime64[D]")


@dataclass(slots=True)
class EventCalendar:
    """Thin helper that aligns an events table to a StockSeries timeline.

    Expected columns in the provided dataframe-like:
    - date: datetime64[D] or parseable to it
    - event_name: str
    Optional columns are preserved in `metadata` but not used in alignment.
    """

    # Stored as NumPy for speed
    dates: np.ndarray  # datetime64[D]
    names: np.ndarray  # object dtype or fixed-width string
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # ---------- Constructors ----------
    @classmethod
    def from_dataframe(cls, df: Any, *, date_col: str = "date", name_col: str = "event_name", keep: Optional[Sequence[str]] = None) -> "EventCalendar":
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pandas is required for EventCalendar.from_dataframe()") from exc
        if not hasattr(df, "__getitem__"):
            raise TypeError("df must be a pandas DataFrame-like with column access")
        d = _to_datetime64D(df[date_col].to_numpy())
        n = df[name_col].astype(str).to_numpy()
        # Deduplicate and sort by date, stable
        order = np.argsort(d)
        d = d[order]
        n = n[order]
        if keep:
            md = {k: df[k].to_numpy()[order] for k in keep if k in df.columns}
        else:
            md = {}
        return cls(dates=d, names=n, metadata=md)

    @classmethod
    def from_csv(cls, path: str, *, date_col: str = "date", name_col: str = "event_name", keep: Optional[Sequence[str]] = None) -> "EventCalendar":
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pandas is required for EventCalendar.from_csv()") from exc
        df = None
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise RuntimeError(f"Failed to read events CSV at {path}") from exc
        return cls.from_dataframe(df, date_col=date_col, name_col=name_col, keep=keep)

    # ---------- Masks & alignment ----------
    def _per_type_day_mask(self, series_days: np.ndarray, event_days: np.ndarray, window: Tuple[int, int]) -> np.ndarray:
        # Base indicator for exact event days
        is_event_day = np.isin(series_days, event_days)
        if window == (0, 0):
            return is_event_day
        l, r = int(window[0]), int(window[1])
        # Expand window using cumulative sum trick
        # Convert bool->int for convolution via cumsum on a rolled window
        x = is_event_day.astype(np.int32)
        # Inclusive window size
        k = r - l + 1
        if k <= 1:
            # Handle cases like (0,0) or (0,1) above will pass here only when k==1, already handled
            return is_event_day
        # Shifted rolling sum using cumsum padding
        pad_left = max(0, -l)
        pad_right = max(0, r)
        x_pad = np.pad(x, (pad_left, pad_right), mode="constant", constant_values=0)
        c = np.cumsum(x_pad)
        # For each index i in original, sum over [i + l + pad_left, i + r + pad_left]
        start = np.arange(x.shape[0]) + l + pad_left
        end = np.arange(x.shape[0]) + r + pad_left
        win_sum = c[end + 1] - c[start]
        return win_sum > 0

    def align(
        self,
        series: Any,
        *,
        window_by_type: Optional[Mapping[str, Tuple[int, int]]] = None,
        types: Optional[Sequence[str]] = None,
        precedence: Optional[Sequence[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Return boolean masks aligned to the series timeline.

        - series: StockSeries (or object exposing as_datetime64())
        - window_by_type: mapping of event_name -> (l_days, r_days) inclusive
        - types: optional whitelist of event names to consider
        - precedence: order to resolve overlaps into a single primary type mask
        Returns a dict with per-type masks and keys: 'any', 'counts'.
        """
        series_days = series.as_datetime64().astype("datetime64[D]")
        unique_types = np.unique(self.names)
        if types is not None:
            mask_types = np.isin(unique_types, np.asarray(types, dtype=str))
            unique_types = unique_types[mask_types]

        per_type_masks: Dict[str, np.ndarray] = {}
        counts = np.zeros(series_days.shape[0], dtype=np.int16)

        for t in unique_types:
            t = str(t)
            t_days = self.dates[self.names == t]
            window = (0, 0) if window_by_type is None else window_by_type.get(t, (0, 0))
            m = self._per_type_day_mask(series_days, t_days, window)
            per_type_masks[t] = m
            counts += m.astype(np.int16)

        any_mask = counts > 0

        # Precedence mask: assign each event day to a single primary type
        if precedence is not None and len(precedence) > 0:
            assigned = np.zeros(series_days.shape[0], dtype=bool)
            for t in precedence:
                m = per_type_masks.get(t)
                if m is None:
                    continue
                per_type_masks[t] = np.where(assigned, False, m)
                assigned = assigned | m
        return {**per_type_masks, "any": any_mask, "counts": counts}

    def any_mask(self, series: Any, window: Tuple[int, int] = (0, 0), types: Optional[Sequence[str]] = None) -> np.ndarray:
        series_days = series.as_datetime64().astype("datetime64[D]")
        if types is None:
            event_days = self.dates
        else:
            sel = np.isin(self.names, np.asarray(types, dtype=str))
            event_days = self.dates[sel]
        return self._per_type_day_mask(series_days, event_days, window)

    # ---------- Introspection helpers ----------
    def unique_event_names(self) -> np.ndarray:
        """Return sorted unique event names."""
        return np.unique(self.names.astype(str))

    def event_dates(
        self,
        event_name: str,
        *,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        as_strings: bool = False,
        unique: bool = True,
        trading_days: Optional[Any] = None,
        series: Optional[Any] = None,
    ) -> np.ndarray:
        """List dates for a specific event name.

        - start/end: optional date bounds (inclusive on both) to filter.
        - as_strings: return ISO strings if True, else datetime64[D] array.
        - unique: drop duplicates (default True).
        - trading_days: optional trading-day filter. Options:
            * True  -> use `series` trading days (requires `series` argument)
            * False/None -> no trading-day filtering
            * StockSeries or iterable of dates -> use those as trading days
        - series: StockSeries to pull trading days from when trading_days is True
        """
        sel = (self.names.astype(str) == str(event_name))
        d = self.dates[sel]
        if start is not None:
            d0 = _to_datetime64D([start])[0]
            d = d[d >= d0]
        if end is not None:
            d1 = _to_datetime64D([end])[0]
            d = d[d <= d1]
        if unique:
            d = np.unique(d)
        if trading_days is not None:
            # Bool flag path: use provided series
            if isinstance(trading_days, (bool, np.bool_)):
                if trading_days:
                    if series is None or not hasattr(series, "as_datetime64"):
                        raise ValueError("When trading_days=True, provide `series` (StockSeries) to derive trading calendar")
                    td = series.as_datetime64().astype("datetime64[D]")
                else:
                    td = None
            else:
                # Accept StockSeries instance or an array-like of dates
                if hasattr(trading_days, "as_datetime64"):
                    td = trading_days.as_datetime64().astype("datetime64[D]")
                else:
                    td = _to_datetime64D(trading_days)

            if td is not None:
                if td.size == 0:
                    d = d[:0]
                else:
                    td_u = np.unique(td)
                    d = d[np.isin(d, td_u)]
        if as_strings:
            return d.astype("datetime64[D]").astype(str)
        return d.astype("datetime64[D]")

    def events_on(self, day: Any) -> np.ndarray:
        """Return event names occurring on a specific calendar day."""
        d = _to_datetime64D([day])[0]
        return self.names[self.dates == d].astype(str)

    @staticmethod
    def to_sigma_t(series: Any, *, sigma_event: float, sigma_calm: float, mask_days: np.ndarray, attribute_to: str = "end") -> np.ndarray:
        """Build per-step sigma_t array from a day mask.

        - mask_days: boolean mask of length series.n, True on event days.
        - attribute_to: 'end' means return on (t-1->t) uses day t mask; 'start' uses day t-1 mask.
        Returns an array of length (series.n - 1) for GBM.paths_timevarying.
        """
        if attribute_to not in ("end", "start"):
            raise ValueError("attribute_to must be 'end' or 'start'")
        if attribute_to == "end":
            step_mask = mask_days[1:]
        else:
            step_mask = mask_days[:-1]
        sigma_t = np.where(step_mask, float(sigma_event), float(sigma_calm))
        return sigma_t.astype(float, copy=False)


__all__ = ["EventCalendar"]


