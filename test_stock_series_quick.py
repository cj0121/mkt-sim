import math
import numpy as np

from stock_series import StockSeries


def main() -> None:
    # 1) Basic from_lists
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    close = [1.0, 1.1, 1.2]
    series = StockSeries.from_lists("SPY", dates=dates, close=close)
    assert series.symbol == "SPY"
    assert series.n == 3
    assert series.close.dtype == np.float64
    assert series.ts_ns.dtype == np.int64
    assert series.as_date_strings() == dates
    ts_ns, px = series.as_arrays()
    assert np.all(np.diff(ts_ns) > 0)
    assert np.allclose(px, np.asarray(close, dtype=np.float64))

    # 2) Utilities: window and slice_by_time
    win = series.window(1, 3)
    assert win.n == 2 and np.allclose(win.close, [1.1, 1.2])
    sliced = series.slice_by_time(series.ts_ns[1], series.ts_ns[2])
    assert sliced.n == 1 and math.isclose(float(sliced.close[0]), 1.1, rel_tol=1e-12)

    # 3) Cached props: log_close and log_returns
    lc = series.log_close
    lr = series.log_returns
    assert lc.shape == (3,)
    assert lr.shape == (2,)
    assert np.allclose(lr, np.diff(np.log(px)))

    # 4) from_records with custom field names
    recs = [
        {"t": "2024-02-01", "c": 10.0, "o": 9.5, "h": 10.1, "l": 9.4, "v": 1000},
        {"t": "2024-02-02", "c": 10.2, "o": 10.0, "h": 10.3, "l": 9.9, "v": 1100},
    ]
    sr = StockSeries.from_records(
        "AAPL",
        records=recs,
        time_field="t",
        close_field="c",
        open_field="o",
        high_field="h",
        low_field="l",
        volume_field="v",
    )
    assert sr.n == 2 and np.allclose(sr.close, [10.0, 10.2])

    # 5) AlphaVantage dict adapter (unsorted dates)
    av_payload = {
        "Meta Data": {},
        "Time Series (Daily)": {
            "2024-01-03": {
                "1. open": "2.0",
                "2. high": "3.0",
                "3. low": "1.9",
                "4. close": "2.1",
                "6. volume": "1000",
            },
            "2024-01-02": {
                "1. open": "1.0",
                "2. high": "2.0",
                "3. low": "0.9",
                "4. close": "1.1",
                "6. volume": "900",
            },
        },
    }
    av_series = StockSeries.from_alphavantage("SPY", av_payload)
    assert av_series.as_date_strings() == ["2024-01-02", "2024-01-03"]
    assert np.allclose(av_series.close, [1.1, 2.1])

    # 6) Polygon dict adapter with ms timestamps (unsorted)
    poly_payload = {
        "results": [
            {"t": 1704326400000, "c": 200.0, "o": 198.0, "h": 201.0, "l": 197.5, "v": 12345},  # 2024-01-04
            {"t": 1704240000000, "c": 198.5, "o": 197.0, "h": 199.0, "l": 196.0, "v": 12000},  # 2024-01-03
        ]
    }
    poly_series = StockSeries.from_polygon("QQQ", poly_payload)
    assert poly_series.as_date_strings() == ["2024-01-03", "2024-01-04"]
    assert np.allclose(poly_series.close, [198.5, 200.0])

    # 7) Invariant: non-strictly-increasing timestamps should fail
    bad_ts = np.asarray([2, 1], dtype=np.int64)
    try:
        _ = StockSeries.from_arrays("BAD", ts_ns=bad_ts, close=[1.0, 2.0])
        raise AssertionError("Expected ValueError for non-increasing ts_ns")
    except ValueError:
        pass

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()


