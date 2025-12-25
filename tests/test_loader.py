import pandas as pd

from optimal_quoting.data.loader import CSVSpec, load_top_of_book_csv


def test_load_top_of_book_csv(tmp_path):
    p = tmp_path / "top.csv"
    pd.DataFrame(
        {
            "timestamp": ["2025-01-01 00:00:00", "2025-01-01 00:00:01"],
            "bid": [100.0, 100.1],
            "ask": [100.2, 100.3],
        }
    ).to_csv(p, index=False)

    spec = CSVSpec(ts_col="timestamp", bid_col="bid", ask_col="ask")
    df = load_top_of_book_csv(p, spec)

    assert len(df) == 2
    assert (df["ask"] > df["bid"]).all()
