import numpy as np

from stock_series import StockSeries
from events import EventCalendar


def main() -> None:
    # Build a tiny StockSeries of 7 consecutive business days
    dates = np.array([
        "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        "2024-01-08", "2024-01-09", "2024-01-10",
    ], dtype="datetime64[D]")
    closes = np.linspace(100.0, 106.0, dates.shape[0])
    s = StockSeries.from_arrays("SPY", ts_ns=dates.astype("datetime64[ns]").view("int64"), close=closes)

    # Synthetic events: two types, overlapping on the same day
    # FOMC on 2024-01-04, CPI on 2024-01-04 and 2024-01-09
    try:
        import pandas as pd  # type: ignore
    except Exception:
        raise SystemExit("pandas required for this quick test")
    df = pd.DataFrame({
        "date": ["2024-01-04", "2024-01-04", "2024-01-09"],
        "event_name": ["FOMC Meeting", "Consumer Price Index", "Consumer Price Index"],
    })
    cal = EventCalendar.from_dataframe(df)

    masks = cal.align(
        s,
        window_by_type={"FOMC Meeting": (0, 0), "Consumer Price Index": (0, 0)},
        precedence=["FOMC Meeting", "Consumer Price Index"],
    )

    any_mask = masks["any"]
    m_fomc = masks["FOMC Meeting"]
    m_cpi = masks["Consumer Price Index"]
    counts = masks["counts"]

    # any: treats multiple events on same day as one
    assert any_mask.tolist() == [False, False, True, False, False, True, False]
    # counts: 2 on 2024-01-04 (both events), 1 on 2024-01-09
    assert counts.tolist() == [0, 0, 2, 0, 0, 1, 0]
    # precedence: day 2024-01-04 assigned to FOMC, not CPI
    assert m_fomc.tolist() == [False, False, True, False, False, False, False]
    assert m_cpi.tolist() == [False, False, False, False, False, True, False]

    # Build sigma_t attributing return to end day
    sigma_t = EventCalendar.to_sigma_t(s, sigma_event=0.3, sigma_calm=0.2, mask_days=any_mask, attribute_to="end")
    # length should be n-1
    assert sigma_t.shape[0] == s.n - 1

    print("EVENTS TESTS PASSED")


if __name__ == "__main__":
    main()


