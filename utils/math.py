import numpy as np


def ewma(arr, lam=0.97):
    """
    Exponentially Weighted Moving Average (EWMA).

    Parameters
    ----------
    arr : np.ndarray
        1D array of values.
    lam : float
        Decay factor in (0,1). Higher lam = longer memory.
        Example: lam=0.97 ~ 2y half-life with daily data.

    Returns
    -------
    np.ndarray
        EWMA-smoothed series, same length as arr.
    """
    arr = np.asarray(arr, dtype=float)
    out = np.empty_like(arr)
    out[0] = arr[0]
    alpha = 1 - lam

    for t in range(1, len(arr)):
        out[t] = lam * out[t-1] + alpha * arr[t]
    return out


def rolling_mean(arr, window: int, *, center: bool = False, min_periods: int = 1):
    """Simple rolling average using cumulative sums (O(n)).

    Parameters
    ----------
    arr : array-like
        1D numeric data.
    window : int
        Window size in elements (> 0).
    center : bool
        If True, center the window on each element (floor((w-1)/2) on both sides).
        If False, use trailing window ending at current index.
    min_periods : int
        Minimum observations in window required to compute a value. Defaults to 1.

    Returns
    -------
    np.ndarray
        Rolling mean of same length as input. Positions with fewer than
        min_periods observations are set to np.nan.
    """
    x = np.asarray(arr, dtype=float)
    n = x.size
    if window <= 0:
        raise ValueError("window must be > 0")
    if n == 0:
        return np.array([], dtype=float)

    if not center:
        # Trailing window: use cumulative sum trick
        c = np.cumsum(np.insert(x, 0, 0.0))  # length n+1
        # sum over [i-window+1, i]
        sums = c[window:] - c[:-window]
        means = sums / float(window)
        # Prefill first window-1 values with partial means if allowed, else NaN
        out = np.empty(n, dtype=float)
        if window > 1:
            # partial sums for first window-1
            pref = c[1:window] / np.arange(1, window)
            out[:window-1] = np.where(np.arange(1, window) >= min_periods, pref, np.nan)
        out[window-1:] = np.where(np.arange(window, n+1) - (window-1) >= min_periods, means, np.nan)
        return out

    # Centered window
    half_left = (window - 1) // 2
    half_right = window - 1 - half_left
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        a = max(0, i - half_left)
        b = min(n, i + half_right + 1)
        count = b - a
        if count >= min_periods:
            s = c[b] - c[a]
            out[i] = s / float(count if count < window else window)
    return out


def rolling_median(arr, window: int, *, center: bool = False, min_periods: int = 1):
    """Rolling median (robust) with optional centered window.

    This implementation prioritizes clarity and correctness. For very large
    windows, consider using pandas or numba for speed.

    Parameters
    ----------
    arr : array-like
        1D numeric data.
    window : int
        Window size (> 0).
    center : bool
        If True, center the window on each index; otherwise trailing.
    min_periods : int
        Minimum observations required to compute a median; else NaN.

    Returns
    -------
    np.ndarray
        Rolling medians, same length as input, dtype float.
    """
    x = np.asarray(arr, dtype=float)
    n = x.size
    if window <= 0:
        raise ValueError("window must be > 0")
    if n == 0:
        return np.array([], dtype=float)

    out = np.full(n, np.nan, dtype=float)

    if not center:
        # Trailing window [i-window+1, i]
        for i in range(n):
            a = max(0, i - window + 1)
            b = i + 1
            count = b - a
            if count >= min_periods:
                out[i] = float(np.median(x[a:b]))
        return out

    # Centered window
    half_left = (window - 1) // 2
    half_right = window - 1 - half_left
    for i in range(n):
        a = max(0, i - half_left)
        b = min(n, i + half_right + 1)
        count = b - a
        if count >= min_periods:
            out[i] = float(np.median(x[a:b]))
    return out