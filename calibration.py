from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any

import os
import numpy as np

from alphavantage_client import AlphaVantageClient


TRADING_DAYS_DEFAULT: int = 252


@dataclass
class CalibrationResult:
    symbol: str
    adjusted: bool
    trading_days: int
    mu: float                 # annual drift for GBM
    sigma: float              # annual volatility for GBM
    mu_log_annual: float      # annualized mean log return
    sigma_daily: float        # daily std of log returns
    mean_daily_log: float     # daily mean of log returns
    num_days_used: int
    start_date: Optional[np.datetime64]
    end_date: Optional[np.datetime64]


class Calibration:
    """Estimate GBM parameters (mu, sigma) from historical daily prices.

    - Defaults to AlphaVantage for data, but accepts any provider with the
      method: fetch_daily_series(symbol, adjusted=True|False, outputsize='full'|'compact')
      returning (dates: np.ndarray[datetime64[D]], closes: np.ndarray[float])
    - You can also bypass APIs with `calibrate_from_series((dates, prices))`.
    """

    def __init__(self, provider: Optional[Any] = None, trading_days: int = TRADING_DAYS_DEFAULT) -> None:
        self.trading_days = int(trading_days)
        if provider is None:
            # Ensure provider is initialized using environment (.env if present)
            self.provider = None
            self.load_provider()
        else:
            self.provider = provider

    def load_provider(self, *, api_key: Optional[str] = None) -> Any:
        """Load/default an API provider and attach to self.provider.

        - Attempts to load environment variables from a .env if dotenv_path is provided.
        - If api_key is None, reads ALPHAVANTAGE_API_KEY from the environment.
        - Returns the instantiated provider.
        """
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv()
        except Exception:
            pass
        key = api_key if api_key is not None else os.getenv("ALPHAVANTAGE_API_KEY")
        self.provider = AlphaVantageClient(api_key=key)
        return self.provider

    def calibrate(
        self,
        symbol: str,
        *,
        adjusted: bool = True,
        start: Optional[Tuple[int, int, int]] = None,  # (YYYY, M, D) or None
        end: Optional[Tuple[int, int, int]] = None,
        outputsize: str = "full",
    ) -> CalibrationResult:
        dates, prices = self._fetch_series(symbol, adjusted=adjusted, outputsize=outputsize)
        return self.calibrate_from_series((dates, prices), symbol=symbol, adjusted=adjusted, start=start, end=end)

    def calibrate_from_series(
        self,
        series: Tuple[np.ndarray, np.ndarray],
        *,
        symbol: str = "",
        adjusted: bool = True,
        start: Optional[Tuple[int, int, int]] = None,
        end: Optional[Tuple[int, int, int]] = None,
    ) -> CalibrationResult:
        """Calibrate from an in-memory series shaped like (dates, closes).

        - dates: np.ndarray[datetime64[D]] sorted ascending
        - closes: np.ndarray[float] same length
        """
        dates, closes = series

        # Filter by date bounds if provided
        if start is not None:
            start_d = np.datetime64(f"{start[0]:04d}-{start[1]:02d}-{start[2]:02d}", "D")
        else:
            start_d = None
        if end is not None:
            end_d = np.datetime64(f"{end[0]:04d}-{end[1]:02d}-{end[2]:02d}", "D")
        else:
            end_d = None

        mask = np.ones(dates.shape[0], dtype=bool)
        if start_d is not None:
            mask &= dates >= start_d
        if end_d is not None:
            mask &= dates <= end_d

        dates_used = dates[mask]
        prices_used = closes[mask]

        # Compute daily log returns x_t = ln(S_t / S_{t-1})
        # Expect ascending daily closes; length must be >= 2
        if prices_used.size < 2:
            raise ValueError("Not enough price points to compute returns (need at least 2)")

        # Drop NaNs if present (minimal hygiene only)
        valid_mask = np.isfinite(prices_used)
        dates_used = dates_used[valid_mask]
        prices_used = prices_used[valid_mask]
        if prices_used.size < 2:
            raise ValueError("Not enough finite price points after filtering")

        log_prices = np.log(prices_used.astype(float))
        daily_log_returns = np.diff(log_prices)

        # Use separated helpers for clarity and reuse
        mu_log_annual, mean_daily_log = self.compute_mu_log_annual(daily_log_returns)
        sigma_annual, sigma_daily = self.compute_sigma_annual(daily_log_returns)
        mu = self.compute_mu(mu_log_annual, sigma_annual)

        return CalibrationResult(
            symbol=symbol,
            adjusted=adjusted,
            trading_days=self.trading_days,
            mu=mu,
            sigma=sigma_annual,
            mu_log_annual=mu_log_annual,
            sigma_daily=sigma_daily,
            mean_daily_log=mean_daily_log,
            num_days_used=int(daily_log_returns.size),
            start_date=dates_used[0] if dates_used.size > 0 else None,
            end_date=dates_used[-1] if dates_used.size > 0 else None,
        )

    # ---- internals ----
    def _fetch_series(self, symbol: str, *, adjusted: bool, outputsize: str) -> Tuple[np.ndarray, np.ndarray]:
        provider = self.provider
        fetch = getattr(provider, "fetch_daily_series", None)
        if not callable(fetch):
            raise TypeError("provider must define fetch_daily_series(symbol, adjusted=True|False, outputsize=...)")
        series = fetch(symbol, adjusted=adjusted, outputsize=outputsize)
        if not isinstance(series, tuple) or len(series) != 2:
            raise ValueError("fetch_daily_series must return (dates, closes)")
        dates, closes = series
        return dates, closes

    # ---- statistics helpers ----
    def compute_sigma_annual(self, daily_log_returns: np.ndarray, ddof: int = 1) -> Tuple[float, float]:
        """Compute annual sigma from daily log returns.

        Returns (sigma_annual, sigma_daily).
        """
        sigma_daily = float(np.std(daily_log_returns, ddof=ddof))
        sigma_annual = float(sigma_daily * np.sqrt(self.trading_days))
        return sigma_annual, sigma_daily

    def compute_mu_log_annual(self, daily_log_returns: np.ndarray) -> Tuple[float, float]:
        """Compute annualized mean log return and the daily mean.

        Returns (mu_log_annual, mean_daily_log).
        """
        mean_daily_log = float(np.mean(daily_log_returns))
        mu_log_annual = float(mean_daily_log * self.trading_days)
        return mu_log_annual, mean_daily_log

    def compute_mu(self, mu_log_annual: float, sigma_annual: float) -> float:
        """Compute GBM drift from annualized log-mean and annual sigma.

        mu = mu_log_annual + 0.5 * sigma_annual^2
        """
        return float(mu_log_annual + 0.5 * (sigma_annual ** 2))


__all__ = ["Calibration", "CalibrationResult", "calibrate_from_series", "calibrate"]


