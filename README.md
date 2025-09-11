## mkt_sim

Lightweight market simulation and data utilities. The project focuses on:
- Fast NumPy-based compute for GBM simulations and analysis
- Clean, standardized ingestion across data providers
- Small, composable helpers for plotting and calibration

---

### Modules at a glance
- `stock_series.py`
  - Thin, slotted, frozen container (`StockSeries`) for a single symbol
  - Canonical schema: `ts_ns:int64` (epoch ns), `close:float64`
  - Optional arrays: `open`, `high`, `low`, `adj_close`, `volume`
  - Constructors: `from_arrays`, `from_lists`, `from_pandas`, `from_records`
  - Provider adapters: `from_alphavantage`, `from_polygon`
  - Utilities: `as_arrays`, `window`, `slice_by_time`, `as_datetime64`, `as_date_strings`
  - Cached: `log_close`, `log_returns`
- `gbm.py`
  - `GBM` simulator class: `step`, `path`, `paths`, `paths_timevarying`
  - Plotting: `plot_paths`, `sample_percentile_paths`, `plot_percentile_paths`,
    `select_terminal_percentile_paths`, `plot_terminal_percentile_paths`
- `calibration.py`
  - `Calibration` utilities to estimate GBM parameters (mu, sigma) from daily data
- `alphavantage_client.py`
  - Minimal client for AlphaVantage: fetch latest price, daily series
- `polygon_client.py`
  - Minimal client for Polygon: fetch latest price, SPX helpers

Notebooks (optional): `eda.ipynb`, `event_scheduler.ipynb`, `template.ipynb`.

Data folders (optional): `data/bls`, `data/fomc`, `data/daily_closing_price`.

---

### Setup

- Environment
```bash
conda create -n mkt_sim python=3.12 -y
conda activate mkt_sim
```

- Core dependencies
```bash
pip install numpy pandas
```

- Optional dependencies
```bash
# Plotting utilities
pip install plotly
# Provider clients
pip install requests python-dotenv
pip install polygon-api-client  # if you plan to use Polygon
# Notebooks
pip install jupyter
```

- Environment variables
  - `ALPHAVANTAGE_API_KEY` for AlphaVantage
  - `POLYGON_API_KEY` for Polygon
  - You can put them in a local `.env` and use `python-dotenv`.

---

### Project layout (excerpt)
```
/Users/honeybunny/Desktop/txt/personal-projects/mkt_sim/
  alphavantage_client.py
  calibration.py
  gbm.py
  polygon_client.py
  stock_series.py
  test_gbm_all.py
  test_stock_series_quick.py
  data/
    bls/
    fomc/
```

---

### StockSeries quickstart
```python
from stock_series import StockSeries

# Build from lists (dates can be ISO strings or datetime64)
series = StockSeries.from_lists(
    symbol="SPY",
    dates=["2024-01-02", "2024-01-03", "2024-01-04"],
    close=[470.0, 472.5, 471.8],
)

# Use in simulations: pass arrays
_, px = series.as_arrays()
# simulate_gbm(px, ...)

# Helpers
dates_str = series.as_date_strings()  # ["2024-01-02", ...]
log_returns = series.log_returns      # cached after first access
```

Provider adapters:
```python
# AlphaVantage dict payload or DataFrame
av_series = StockSeries.from_alphavantage("SPY", payload)

# Polygon aggregates (/v2/aggs) JSON dict
poly_series = StockSeries.from_polygon("QQQ", payload)
```

---

### GBM simulation quickstart
```python
import numpy as np
from gbm import GBM

S0, mu, sigma = 470.0, 0.08, 0.20
sim = GBM(trading_days=252)

# Single path for 1Y
prices = sim.path(S0, mu, sigma, n_days=252, seed=42)

# Many paths
paths = sim.paths(S0, mu, sigma, n_days=252, n_paths=10000, seed=42)

# Time-varying params via arrays
mu_t = np.full(252, mu)
sigma_t = np.full(252, sigma)
# e.g., higher vol in last quarter
sigma_t[-63:] = 0.30
paths_tv = sim.paths_timevarying(S0, mu_t=mu_t, sigma_t=sigma_t, n_paths=1000, seed=42)
```

Plotting helpers (optional, requires `plotly`):
```python
from gbm import plot_paths, sample_percentile_paths, plot_percentile_paths

fig = plot_paths(paths, max_paths_to_plot=500)
# fig.show()

pp = sample_percentile_paths(paths, percentiles=(5, 25, 50, 75, 95))
fig2 = plot_percentile_paths(pp)
# fig2.show()
```

---

### Calibration quickstart
Estimate GBM parameters from daily closes.
```python
import numpy as np
from calibration import Calibration

cal = Calibration()  # uses AlphaVantage if env var is set
res = cal.calibrate(symbol="SPY", adjusted=True, outputsize="full")
print(res.mu, res.sigma)

# From an in-memory series (dates: datetime64[D], closes: float)
dates = np.array(["2024-01-02", "2024-01-03"], dtype="datetime64[D]")
closes = np.array([470.0, 472.5], dtype=float)
res2 = cal.calibrate_from_series((dates, closes), symbol="SPY", adjusted=True)
```

Key formulas (annualized):
- `sigma_annual = std(daily_log_returns) * sqrt(trading_days)`
- `mu_log_annual = mean(daily_log_returns) * trading_days`
- `mu = mu_log_annual + 0.5 * sigma_annual^2`

---

### Data clients
AlphaVantage client:
```python
from alphavantage_client import AlphaVantageClient

av = AlphaVantageClient()
price = av.fetch_price("SPY")
series = av.fetch_daily_series("SPY", adjusted=True, outputsize="compact")  # (dates, closes)
```

Polygon client:
```python
from polygon_client import PolygonClient

pg = PolygonClient()
price = pg.fetch_price("SPY")
spx  = pg.fetch_spx()
```

---

### Tests
Quick sanity test for StockSeries:
```bash
python test_stock_series_quick.py
```
GBM tests (if present):
```bash
python test_gbm_all.py
```

Expected quick test output:
```
ALL TESTS PASSED
```

---

### Design notes
- Canonical time is stored as `int64` epoch nanoseconds for speed and clarity (no strings in hot paths)
- Heavy math remains NumPy-native; data containers are thin boundaries
- Use pandas only at ingestion/IO boundaries; keep compute paths on NumPy arrays

---

### AlphaVantage client usage
- Setup
  - Install: `pip install requests python-dotenv`
  - Set env: `export ALPHAVANTAGE_API_KEY=YOUR_KEY` (or create a `.env` with `ALPHAVANTAGE_API_KEY=...`)

- Instantiate
```python
from alphavantage_client import AlphaVantageClient

av = AlphaVantageClient()               # auto-reads ALPHAVANTAGE_API_KEY
av = AlphaVantageClient(api_key="...")  # explicit key
```

- Latest price
```python
price = av.fetch_price("SPY")  # float | None
if price is None:
    print("No price (rate limit or network issue)")
```

- Daily series
```python
# Returns (dates: np.ndarray[datetime64[D]], closes: np.ndarray[float])
dates, closes = av.fetch_daily_series("SPY", adjusted=True, outputsize="compact")

# Full history (slower; observe rate limits)
# dates, closes = av.fetch_daily_series("SPY", adjusted=True, outputsize="full")
```

- Resolve S0 helper
```python
from alphavantage_client import resolve_S0
S0 = resolve_S0(S0=None, symbol="SPY")  # tries live/intraday/daily in order
```

- Convert to StockSeries
```python
from stock_series import StockSeries
ts_ns = dates.astype("datetime64[ns]").view("int64")
series = StockSeries.from_arrays("SPY", ts_ns=ts_ns, close=closes)
```

- Rate limits & retries
  - Free tier commonly ~5 req/min and daily caps. Expect occasional JSON notes.
  - The client retries (`max_retries`) and returns `None` on failure; check for `None` and back off (`time.sleep(12)` between calls on free tier).
  - Tune: `AlphaVantageClient(timeout_sec=10.0, max_retries=2)`.

---

### License
TBD

### Events: EventCalendar
```python
import pandas as pd
from events import EventCalendar
from stock_series import StockSeries
from gbm import GBM

# Load price series (dates, closes) -> StockSeries
series = StockSeries.from_alphavantage("SPY", (dates, closes))

# Load events from CSV/DataFrame
cal = EventCalendar.from_csv("data/events_calendar.csv")

# Build per-type and combined masks aligned to the series
masks = cal.align(
    series,
    window_by_type={
        "FOMC Meeting": (0, 0),
        "Consumer Price Index": (0, 0),
        "Employment Situation": (0, 0),
    },
    precedence=["FOMC Meeting", "Consumer Price Index", "Employment Situation"],
)

# Combined mask: multiple events on same day -> one True
event_days = masks["any"]            # shape: (series.n,)
counts = masks["counts"]             # number of events per day
fomc_days = masks["FOMC Meeting"]    # precedence-assigned mask

# Build per-step sigma_t attributing return to end day
sigma_t = EventCalendar.to_sigma_t(series, sigma_event=0.30, sigma_calm=0.20, mask_days=event_days, attribute_to="end")

# Simulate with time-varying sigma
gbm = GBM()
paths = gbm.paths_timevarying(S0=series.close[-1], mu_t=np.full(series.n - 1, 0.08), sigma_t=sigma_t, n_paths=10000, seed=42)
```
