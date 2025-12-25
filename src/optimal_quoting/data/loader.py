from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CSVSpec:
    """
    Standardizes raw CSV into a canonical dataframe.

    Expected output columns:
      - ts (datetime64[ns], tz-naive)
      - bid, ask (float)
      - bid_size, ask_size (float, optional)
    """

    ts_col: str = "timestamp"
    bid_col: str = "bid"
    ask_col: str = "ask"
    bid_size_col: str | None = None
    ask_size_col: str | None = None


def load_top_of_book_csv(path: str | Path, spec: CSVSpec) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    if spec.ts_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {spec.ts_col}")

    out = pd.DataFrame()
    out["ts"] = pd.to_datetime(df[spec.ts_col], utc=False)

    for col_name, out_name in [(spec.bid_col, "bid"), (spec.ask_col, "ask")]:
        if col_name not in df.columns:
            raise ValueError(f"Missing column: {col_name}")
        out[out_name] = pd.to_numeric(df[col_name], errors="raise")

    if spec.bid_size_col and spec.bid_size_col in df.columns:
        out["bid_size"] = pd.to_numeric(df[spec.bid_size_col], errors="raise")
    if spec.ask_size_col and spec.ask_size_col in df.columns:
        out["ask_size"] = pd.to_numeric(df[spec.ask_size_col], errors="raise")

    # Basic sanity
    if (out["ask"] <= out["bid"]).any():
        bad = out.index[out["ask"] <= out["bid"]][0]
        raise ValueError(f"Found ask <= bid at row {bad}")

    out = out.sort_values("ts").reset_index(drop=True)
    return out
