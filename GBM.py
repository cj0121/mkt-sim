import math
from typing import Optional, Dict, Sequence, Union, Tuple
from datetime import date as _date
from dataclasses import dataclass

import numpy as np
import plotly.express as px


class GBM:
    """Geometric Brownian Motion simulator.

    This class encapsulates common GBM utilities shown in the notebook:
    - One-step update (single shock)
    - Single-path simulation over n_days
    - Multi-path simulation with constant parameters
    - Multi-path simulation with time-varying parameters

    All drift (mu) and volatility (sigma) parameters are annualized.
    Time is discretized into trading days with dt = 1 / 252 by default.
    """

    def __init__(
        self,
        trading_days: int = 252,
        start_date: Optional[Union[str, np.datetime64, _date]] = None,
        holidays: Optional[Sequence[Union[str, np.datetime64, _date]]] = None,
    ) -> None:
        self.trading_days = trading_days
        self.dt = 1.0 / float(trading_days)
        self.sqrt_dt = math.sqrt(self.dt)

        # Dates and business day calendar
        self.start_date = self._to_datetime64D(start_date) if start_date is not None else np.datetime64("today", "D")
        holidays_arr = None
        if holidays is not None and len(holidays) > 0:
            holidays_arr = np.array([self._to_datetime64D(h) for h in holidays], dtype="datetime64[D]")
        self._busdaycal = np.busdaycalendar(weekmask="1111100", holidays=holidays_arr)

    # ---- Date helpers ----
    @staticmethod
    def _to_datetime64D(d: Union[str, np.datetime64, _date]) -> np.datetime64:
        if isinstance(d, np.datetime64):
            return d.astype("datetime64[D]")
        if isinstance(d, _date):
            return np.datetime64(d.isoformat(), "D")
        # assume ISO string
        return np.datetime64(str(d), "D")

    def _gen_dates_by_n(self, n_days: int) -> np.ndarray:
        # Produce n_days+1 business dates starting at self.start_date
        offsets = np.arange(n_days + 1, dtype=int)
        return np.busday_offset(self.start_date, offsets, roll="forward", busdaycal=self._busdaycal)

    def _count_days_until(self, end_date: Union[str, np.datetime64, _date]) -> int:
        end_d = self._to_datetime64D(end_date)
        # number of business days between start and end (end exclusive)
        return int(np.busday_count(self.start_date, end_d, busdaycal=self._busdaycal))

    def step(self, S_t: float, mu: float, sigma: float, Z: Optional[float] = None, seed: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> float:
        """Advance one step of length dt from price S_t using GBM.

        If Z is not provided, a standard normal draw is produced using the provided seed.
        """
        # Basic validations
        if not np.isfinite(S_t):
            raise ValueError("S_t must be finite")
        if not np.isfinite(mu):
            raise ValueError("mu must be finite")
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and >= 0")
        if Z is None:
            if rng is None:
                rng = np.random.default_rng(seed)
            Z = float(rng.standard_normal())

        drift_term = (mu - 0.5 * sigma**2) * self.dt
        diffusion_term = sigma * self.sqrt_dt * Z
        return float(S_t * math.exp(drift_term + diffusion_term))

    def path(
        self,
        S0: float,
        mu: float,
        sigma: float,
        n_days: Optional[int] = 252,
        *,
        end_date: Optional[Union[str, np.datetime64, _date]] = None,
        seed: Optional[int] = None,
        return_dates: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Simulate a single GBM path of length n_days + 1 (including S0).

        Returns an array of shape (n_days + 1,) with prices [S_0, S_1, ..., S_n].
        """
        if end_date is not None:
            n_days = self._count_days_until(end_date)
        assert n_days is not None

        if n_days < 1:
            raise ValueError("n_days must be >= 1")
        if not np.isfinite(S0):
            raise ValueError("S0 must be finite")
        if not np.isfinite(mu):
            raise ValueError("mu must be finite")
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and >= 0")

        rng = np.random.default_rng(seed)
        Zs = rng.standard_normal(n_days)

        prices = np.empty(n_days + 1, dtype=float)
        prices[0] = S0

        drift = (mu - 0.5 * sigma**2) * self.dt
        diff_scale = sigma * self.sqrt_dt

        # Stable vectorized approach via cumulative log-returns
        r_log = drift + diff_scale * Zs  # (n_days,)
        cum_log = np.cumsum(r_log)
        growth = np.exp(cum_log)  # (n_days,)
        prices[1:] = S0 * growth
        if return_dates:
            dates = self._gen_dates_by_n(n_days)
            return prices, dates
        return prices

    def paths(
        self,
        S0: float,
        mu: float,
        sigma: float,
        n_days: Optional[int] = 252,
        n_paths: int = 100_000,
        *,
        end_date: Optional[Union[str, np.datetime64, _date]] = None,
        seed: Optional[int] = None,
        return_dates: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Simulate multiple GBM paths with constant mu and sigma.

        Returns a matrix of shape (n_paths, n_days + 1).
        """
        if end_date is not None:
            n_days = self._count_days_until(end_date)
        assert n_days is not None

        if n_days < 1:
            raise ValueError("n_days must be >= 1")
        if n_paths < 1:
            raise ValueError("n_paths must be >= 1")
        if not np.isfinite(S0):
            raise ValueError("S0 must be finite")
        if not np.isfinite(mu):
            raise ValueError("mu must be finite")
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and >= 0")

        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n_paths, n_days))

        drift = (mu - 0.5 * sigma**2) * self.dt
        diff = sigma * self.sqrt_dt

        r_log = drift + diff * Z  # (n_paths, n_days)
        cum_log = np.cumsum(r_log, axis=1)
        growth = np.exp(cum_log)  # (n_paths, n_days)

        paths = np.empty((n_paths, n_days + 1), dtype=float)
        paths[:, 0] = S0
        paths[:, 1:] = S0 * growth
        if return_dates:
            dates = self._gen_dates_by_n(n_days)
            return paths, dates
        return paths

    def paths_timevarying(
        self,
        S0: float,
        mu_t: Optional[np.ndarray] = None,
        sigma_t: Optional[np.ndarray] = None,
        n_paths: int = 1,
        *,
        base_mu: Optional[float] = None,
        base_sigma: Optional[float] = None,
        overrides: Optional[Sequence[Tuple[Union[str, np.datetime64, _date], Union[str, np.datetime64, _date], float, float]]] = None,
        n_days: Optional[int] = None,
        end_date: Optional[Union[str, np.datetime64, _date]] = None,
        seed: Optional[int] = None,
        return_dates: bool = False,
        keepdims: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Simulate GBM paths with time-varying daily parameters.

        Modes:
        - Array mode: provide `mu_t` and `sigma_t` of length n_days.
        - Date override mode: provide `base_mu`, `base_sigma`, optional `overrides`, and either `n_days` or `end_date`.

        Overrides semantics: parameters apply to the step starting on each business date in dates[:-1].

        If n_paths == 1, returns a single path shaped (n_days + 1,).
        Otherwise returns a matrix shaped (n_paths, n_days + 1).
        Use keepdims=True to always return 2D when n_paths == 1.
        """
        # Determine n_days and date index when needed
        dates: Optional[np.ndarray] = None

        if mu_t is not None and sigma_t is not None:
            n_days_calc = int(len(mu_t))
            if len(sigma_t) != n_days_calc:
                raise ValueError("mu_t and sigma_t must have the same length")
            # Validate arrays
            mu_t = np.asarray(mu_t)
            sigma_t = np.asarray(sigma_t)
            if mu_t.ndim != 1 or sigma_t.ndim != 1:
                raise ValueError("mu_t and sigma_t must be 1-D arrays")
            if not np.all(np.isfinite(mu_t)):
                raise ValueError("mu_t contains non-finite values")
            if not np.all(np.isfinite(sigma_t)) or not np.all(sigma_t >= 0):
                raise ValueError("sigma_t must be finite and >= 0")
            n_days = n_days_calc
        else:
            if end_date is not None:
                n_days = self._count_days_until(end_date)
            if n_days is None:
                raise ValueError("Provide either (mu_t and sigma_t) or (base params and n_days/end_date)")
            if n_days < 1:
                raise ValueError("n_days must be >= 1")
            # Build per-day params from base + overrides across business dates
            if base_mu is None or base_sigma is None:
                raise ValueError("base_mu and base_sigma are required when mu_t/sigma_t are not provided")
            if not np.isfinite(base_mu):
                raise ValueError("base_mu must be finite")
            if not np.isfinite(base_sigma) or base_sigma < 0:
                raise ValueError("base_sigma must be finite and >= 0")
            dates = self._gen_dates_by_n(n_days)
            step_dates = dates[:-1]
            mu_t = np.full(n_days, float(base_mu), dtype=float)
            sigma_t = np.full(n_days, float(base_sigma), dtype=float)
            if overrides:
                for (d_start, d_end, mu_o, sigma_o) in overrides:
                    start_d = self._to_datetime64D(d_start)
                    end_d = self._to_datetime64D(d_end)
                    mask = (step_dates >= start_d) & (step_dates <= end_d)
                    if not np.isfinite(mu_o):
                        raise ValueError("override mu must be finite")
                    if not np.isfinite(sigma_o) or sigma_o < 0:
                        raise ValueError("override sigma must be finite and >= 0")
                    mu_t[mask] = float(mu_o)
                    sigma_t[mask] = float(sigma_o)

        # Now simulate
        rng = np.random.default_rng(seed)
        drift = (mu_t - 0.5 * sigma_t**2) * self.dt  # type: ignore[operator]
        diff = sigma_t * self.sqrt_dt                # type: ignore[operator]

        if n_paths == 1:
            Z = rng.standard_normal(n_days)  # type: ignore[arg-type]
            r_log = drift + diff * Z  # type: ignore[operator]
            cum_log = np.cumsum(r_log)
            growth = np.exp(cum_log)
            if keepdims:
                prices = np.empty((1, n_days + 1), dtype=float)  # type: ignore[arg-type]
                prices[:, 0] = S0
                prices[:, 1:] = S0 * growth
            else:
                prices = np.empty(n_days + 1, dtype=float)  # type: ignore[arg-type]
                prices[0] = S0
                prices[1:] = S0 * growth
            if return_dates:
                if dates is None:
                    dates = self._gen_dates_by_n(n_days)  # type: ignore[arg-type]
                return prices, dates
            return prices

        Z = rng.standard_normal((n_paths, n_days))  # type: ignore[arg-type]
        r_log = drift[np.newaxis, :] + diff[np.newaxis, :] * Z  # type: ignore[index]
        cum_log = np.cumsum(r_log, axis=1)
        growth = np.exp(cum_log)
        paths = np.empty((n_paths, n_days + 1), dtype=float)
        paths[:, 0] = S0
        paths[:, 1:] = S0 * growth
        if return_dates:
            if dates is None:
                dates = self._gen_dates_by_n(n_days)  # type: ignore[arg-type]
            return paths, dates
        return paths


__all__ = [
    "GBM",
    "plot_paths",
    "PercentilePaths",
    "sample_percentile_paths",
    "plot_percentile_paths",
    "TerminalPercentileSelection",
    "select_terminal_percentile_paths",
    "plot_terminal_percentile_paths",
]


def plot_paths(paths: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], max_paths_to_plot: int = 1_000, seed: Optional[int] = 0, title: Optional[str] = None, width: Optional[int] = 800, height: Optional[int] = 600):
    """Plot GBM paths using plotly express with optional sampling.

    - paths: array shaped (n_paths, n_steps) or (n_steps,) for a single path.
    - max_paths_to_plot: cap on number of paths to display.
    - seed: RNG seed for reproducible subsampling when needed.
    - title: optional plot title.
    """
    if paths is None:
        raise ValueError("paths must not be None")

    dates: Optional[np.ndarray] = None
    if isinstance(paths, tuple):
        paths_arr, dates = paths
    else:
        paths_arr = paths

    # Normalize to 2D: (n_paths, n_steps)
    if paths_arr.ndim == 1:
        paths_to_use = paths_arr.reshape(1, -1)
    elif paths_arr.ndim == 2:
        paths_to_use = paths_arr
    else:
        raise ValueError("paths must be 1D or 2D array")

    n_paths, n_steps = paths_to_use.shape

    # Sampling helper
    def _sample_indices(num: int, k: int, rng_seed: Optional[int]) -> np.ndarray:
        if k >= num:
            return np.arange(num)
        rng = np.random.default_rng(rng_seed)
        return rng.choice(num, size=k, replace=False)

    sel = _sample_indices(n_paths, int(max_paths_to_plot), seed)
    sampled = paths_to_use[sel]

    # Build long-form dataframe for px
    # x: time index 0..n_steps-1, y: price, path_id: selected path index
    time_index = dates if dates is not None else np.arange(n_steps)
    # Repeat time index for each path, and flatten sampled values
    x_vals = np.tile(time_index, sampled.shape[0])
    y_vals = sampled.reshape(-1)
    path_ids = np.repeat(np.arange(sampled.shape[0]), n_steps)

    fig = px.line(x=x_vals, y=y_vals, color=path_ids.astype(str), labels={"x": ("date" if dates is not None else "step"), "y": "price", "color": "path"}, title=title or "GBM Paths", width=width, height=height)
    return fig


@dataclass
class PercentilePaths:
    time: np.ndarray
    percentiles: Dict[int, np.ndarray]  # key: percentile (e.g., 1, 5, 25, ...), value: series over time
    selected_indices: np.ndarray        # indices of sampled paths from original matrix
    n_paths: int
    n_steps: int
    dates: Optional[np.ndarray] = None  # optional business-date axis (n_steps,)


def sample_percentile_paths(
    paths: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    percentiles: Sequence[int] = (1, 5, 25, 50, 75, 92, 99),
    max_paths_to_sample: int = 100_000,
    seed: Optional[int] = 0,
) -> PercentilePaths:
    """Return percentile summary lines over time from a (possibly huge) set of paths.

    The function optionally subsamples up to max_paths_to_sample paths to compute percentiles.
    - paths: (n_paths, n_steps) or (n_steps,) array
    - percentiles: list/tuple of percentiles to compute, in [0, 100]
    - max_paths_to_sample: cap on number of paths used for computation
    - seed: RNG seed for reproducible subsampling
    """
    if paths is None:
        raise ValueError("paths must not be None")

    dates: Optional[np.ndarray] = None
    if isinstance(paths, tuple):
        paths_arr, dates = paths
    else:
        paths_arr = paths

    # Normalize to 2D
    if paths_arr.ndim == 1:
        paths2d = paths_arr.reshape(1, -1)
    elif paths_arr.ndim == 2:
        paths2d = paths_arr
    else:
        raise ValueError("paths must be 1D or 2D array")

    n_paths, n_steps = paths2d.shape

    # Subsample indices
    k = int(max_paths_to_sample)
    if k >= n_paths:
        sel = np.arange(n_paths)
    else:
        rng = np.random.default_rng(seed)
        sel = rng.choice(n_paths, size=k, replace=False)

    sampled = paths2d[sel]

    # Compute percentiles over axis=0 (time)
    qs = np.asarray(percentiles, dtype=float)
    pct_values = np.percentile(sampled, q=qs, axis=0)
    # Ensure 2D shape (len(percentiles), n_steps)
    if pct_values.ndim == 1:
        pct_values = pct_values.reshape(1, -1)

    pct_map: Dict[int, np.ndarray] = {int(q): pct_values[i] for i, q in enumerate(qs)}
    time = dates if dates is not None else np.arange(n_steps)

    return PercentilePaths(
        time=time,
        percentiles=pct_map,
        selected_indices=sel,
        n_paths=n_paths,
        n_steps=n_steps,
        dates=dates,
    )


def plot_percentile_paths(pp: PercentilePaths, title: Optional[str] = None, width: Optional[int] = 800, height: Optional[int] = 600):
    """Plot percentile summary lines returned by sample_percentile_paths.

    - pp: PercentilePaths object
    - title: optional plot title
    - width/height: figure size
    """
    if pp is None:
        raise ValueError("Percentile summary must not be None")

    # Sort percentiles for consistent legend ordering
    keys = sorted(pp.percentiles.keys())

    # Long-form arrays
    x_vals_list = []
    y_vals_list = []
    label_list = []

    for k in keys:
        series = pp.percentiles[k]
        x_vals_list.append(pp.time)
        y_vals_list.append(series)
        label_list.append(np.full_like(pp.time, fill_value=k, dtype=int))

    x_vals = np.concatenate(x_vals_list)
    y_vals = np.concatenate(y_vals_list)
    labels = np.concatenate(label_list).astype(str)

    fig = px.line(
        x=x_vals,
        y=y_vals,
        color=labels,
        labels={"x": ("date" if (pp.dates is not None or (isinstance(pp.time, np.ndarray) and np.issubdtype(pp.time.dtype, np.datetime64))) else "step"), "y": "price", "color": "percentile"},
        title=title or "GBM Percentile Paths",
        width=width,
        height=height,
    )
    return fig


@dataclass
class TerminalPercentileSelection:
    time: np.ndarray
    selected_indices: np.ndarray            # indices of chosen paths
    selected_percentiles: Dict[int, float]  # percentile -> target terminal value
    selected_paths: np.ndarray              # (k, n_steps) matrix of chosen paths
    n_paths: int
    n_steps: int


def select_terminal_percentile_paths(
    paths: np.ndarray,
    percentiles: Sequence[int] = (1, 5, 25, 50, 75, 95, 99),
    max_paths_to_search: int = 1_000_000,
    seed: Optional[int] = 0,
) -> TerminalPercentileSelection:
    """Select actual paths whose terminal values are closest to specified percentiles.

    - paths: (n_paths, n_steps) or (n_steps,) array
    - percentiles: percentiles in [0, 100]
    - max_paths_to_search: optional subsample size for search scalability
    - seed: RNG seed for reproducible subsampling when needed
    """
    if paths is None:
        raise ValueError("paths must not be None")

    if paths.ndim == 1:
        paths2d = paths.reshape(1, -1)
    elif paths.ndim == 2:
        paths2d = paths
    else:
        raise ValueError("paths must be 1D or 2D array")

    n_paths, n_steps = paths2d.shape

    # Determine the candidate set to search
    if max_paths_to_search >= n_paths:
        search_idx = np.arange(n_paths)
    else:
        rng = np.random.default_rng(seed)
        search_idx = rng.choice(n_paths, size=int(max_paths_to_search), replace=False)

    terminals = paths2d[search_idx, -1]

    # Percentile targets computed on the candidate set's terminal distribution
    qs = np.asarray(percentiles, dtype=float)
    targets = np.percentile(terminals, q=qs)

    # For each target, greedily pick the index with minimal absolute deviation, ensuring uniqueness
    chosen_local = []
    used = np.zeros(terminals.shape[0], dtype=bool)
    for tval in targets:
        diffs = np.abs(terminals - tval)
        # Mask out used candidates by setting their diffs to +inf (avoid 0*inf -> NaN)
        diffs_masked = np.where(used, np.inf, diffs)
        local_idx = int(np.argmin(diffs_masked))
        used[local_idx] = True
        chosen_local.append(local_idx)

    # Map back to original indices; keep unique order by percentile
    chosen_search_idx = np.array(chosen_local, dtype=int)
    chosen_global_idx = search_idx[chosen_search_idx]

    # It's possible multiple percentiles pick the same path (ties). Deduplicate but keep mapping.
    # We'll keep the order aligned with percentiles for plotting/legend consistency.
    selected_paths = paths2d[chosen_global_idx]

    pct_to_target: Dict[int, float] = {int(q): float(targets[i]) for i, q in enumerate(qs)}
    time = np.arange(n_steps)

    return TerminalPercentileSelection(
        time=time,
        selected_indices=chosen_global_idx,
        selected_percentiles=pct_to_target,
        selected_paths=selected_paths,
        n_paths=n_paths,
        n_steps=n_steps,
    )


def plot_terminal_percentile_paths(tp: TerminalPercentileSelection, title: Optional[str] = None, width: Optional[int] = 800, height: Optional[int] = 600):
    """Plot actual paths selected by terminal-value percentiles.

    Each path is labeled by its percentile (e.g., 50 for median terminal value).
    """
    if tp is None:
        raise ValueError("TerminalPercentileSelection must not be None")

    k = tp.selected_paths.shape[0]
    time_index = tp.time
    n_steps = tp.n_steps

    x_vals = np.tile(time_index, k)
    y_vals = tp.selected_paths.reshape(-1)

    # Labels in the same order as input percentiles/selection
    pcts = list(tp.selected_percentiles.keys())
    # Ensure length matches k; if not, truncate or repeat based on selection shape
    if len(pcts) != k:
        pcts = pcts[:k]
    labels = np.repeat([str(p) for p in pcts], n_steps)

    fig = px.line(
        x=x_vals,
        y=y_vals,
        color=labels,
        labels={"x": "step", "y": "price", "color": "terminal percentile"},
        title=title or "GBM Terminal Percentile Paths",
        width=width,
        height=height,
    )
    return fig


# Polygon integration was moved to `polygon_client.py` to keep GBM decoupled from external APIs.
