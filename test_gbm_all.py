import numpy as np
from dotenv import load_dotenv
load_dotenv()

from GBM import (
    GBM,
    plot_paths,
    sample_percentile_paths,
    plot_percentile_paths,
    select_terminal_percentile_paths,
    plot_terminal_percentile_paths,
)


def main():
    gbm = GBM(start_date="2025-01-02", holidays=["2025-01-20"])

    s1 = gbm.step(5000.0, mu=0.06, sigma=0.18, seed=123)
    print("One-step price:", s1)

    path_10, dates_10 = gbm.path(5000.0, 0.06, 0.18, n_days=10, return_dates=True, seed=42)
    print("Single path (10 days) shape:", path_10.shape, "dates:", dates_10[:3], "...", dates_10[-1])

    paths_const, dates_const = gbm.paths(
        5000.0, 0.06, 0.18, n_days=252, n_paths=5000, return_dates=True, seed=1
    )
    print("Const paths shape:", paths_const.shape)

    fig_paths = plot_paths((paths_const, dates_const), max_paths_to_plot=200, seed=7, title="Constant-Param Paths (sample)")
    fig_paths.show()

    pp = sample_percentile_paths(
        (paths_const, dates_const),
        percentiles=(1, 5, 25, 50, 75, 95, 99),
        max_paths_to_sample=2000,
        seed=7,
    )
    print("Percentile series keys:", sorted(pp.percentiles.keys()))
    fig_pct = plot_percentile_paths(pp, title="Percentile Lines (dated)")
    fig_pct.show()

    sel = select_terminal_percentile_paths(
        paths_const,
        percentiles=(1, 5, 25, 50, 75, 95, 99),
        max_paths_to_search=2000,
        seed=7,
    )
    print("Terminal selected indices:", sel.selected_indices[:10], "...")
    fig_term = plot_terminal_percentile_paths(sel, title="Terminal Percentile Paths")
    fig_term.show()

    overrides = [
        ("2025-06-10", "2025-06-17", 0.04, 0.28),
        ("2025-09-01", "2025-09-15", 0.03, 0.22),
    ]
    paths_tv, dates_tv = gbm.paths_timevarying(
        5000.0,
        base_mu=0.06,
        base_sigma=0.18,
        overrides=overrides,
        end_date="2025-12-31",
        n_paths=4000,
        return_dates=True,
        seed=11,
    )
    print("Time-varying paths shape:", paths_tv.shape)

    fig_tv = plot_paths((paths_tv, dates_tv), max_paths_to_plot=150, seed=3, title="Time-Varying Paths (sample)")
    fig_tv.show()

    pp_tv = sample_percentile_paths(
        (paths_tv, dates_tv),
        percentiles=(5, 50, 95),
        max_paths_to_sample=1500,
        seed=3,
    )
    fig_pct_tv = plot_percentile_paths(pp_tv, title="Time-Varying Percentile Lines (dated)")
    fig_pct_tv.show()

    sel_tv = select_terminal_percentile_paths(
        paths_tv,
        percentiles=(5, 50, 95),
        max_paths_to_search=1500,
        seed=3,
    )
    fig_term_tv = plot_terminal_percentile_paths(sel_tv, title="Time-Varying Terminal Percentile Paths")
    fig_term_tv.show()

    n_days_arr = 126
    mu_t = np.full(n_days_arr, 0.06)
    sigma_t = np.full(n_days_arr, 0.18)
    sigma_t[20:30] = 0.30
    path_arrmode, dates_arrmode = gbm.paths_timevarying(
        5000.0, mu_t=mu_t, sigma_t=sigma_t, n_paths=1, return_dates=True, seed=21
    )
    print("Array-mode single path shape:", path_arrmode.shape)
    fig_arrmode = plot_paths((path_arrmode, dates_arrmode), title="Array-Mode Single Path")
    fig_arrmode.show()

    # # 9) Polygon helpers (optional). If module or key missing, skip gracefully.
    # try:
    #     from polygon_client import PolygonClient, fetch_price_polygon, fetch_spx_price_polygon, resolve_S0
    #     import os
    #     key = os.getenv("POLYGON_API_KEY")
    #     pc = PolygonClient(api_key=key) if key else PolygonClient(api_key=None)
    #     spx = pc.fetch_spx()
    #     aapl = pc.fetch_price("AAPL")
    #     print("Polygon fetched I:SPX:", spx, "AAPL:", aapl)
    #     s0_auto = pc.resolve_S0(None, ticker="I:SPX")
    #     print("Resolved S0 (I:SPX):", s0_auto)
    # except Exception as e:
    #     print("Polygon test skipped:", e)

    # 10) AlphaVantage helpers (optional). If key missing, skip gracefully.
    try:
        from alphavantage_client import AlphaVantageClient
        import os
        av_key = os.getenv("ALPHAVANTAGE_API_KEY")
        ac = AlphaVantageClient(api_key=av_key)
        aapl_av = ac.fetch_price("AAPL")
        spy_av = ac.fetch_price("SPY")
        print("AlphaVantage fetched AAPL:", aapl_av, "SPY:", spy_av)
        s0_av = ac.resolve_S0(None, symbol="SPY")
        print("Resolved S0 via AV (SPY):", s0_av)
    except Exception as e:
        print("AlphaVantage test skipped:", e)


if __name__ == "__main__":
    main()


