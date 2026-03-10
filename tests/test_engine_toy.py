from optimal_quoting.backtest.engine import MMParams, run_mm_toy


def test_run_mm_toy_runs():
    p = MMParams(
        dt=1.0,
        T=10.0,
        mid0=100.0,
        sigma=0.01,
        A=1.0,
        k=1.0,
        base_spread=0.2,
        phi=0.0,
        order_size=0.01,
        fee_bps=1.0,
        seed=123,
    )
    df = run_mm_toy(p)
    assert len(df) > 5
    assert "equity" in df.columns
