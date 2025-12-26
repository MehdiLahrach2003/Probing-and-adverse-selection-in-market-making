from __future__ import annotations

import numpy as np
import pandas as pd


def build_intensity_dataset_from_mm(df: pd.DataFrame, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (delta, n) arrays from the toy market-making dataframe.

    We use both sides:
      delta_bid = mid - bid
      delta_ask = ask - mid
      n_bid = 1(fill_bid), n_ask = 1(fill_ask)

    Returns
    -------
    delta : array (2T,)
    n : array (2T,)
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")

    required = {"mid", "bid", "ask", "fill_bid", "fill_ask"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {sorted(missing)}")

    delta_bid = (df["mid"] - df["bid"]).to_numpy(dtype=float)
    delta_ask = (df["ask"] - df["mid"]).to_numpy(dtype=float)

    if (delta_bid < 0).any() or (delta_ask < 0).any():
        raise ValueError("Found negative deltas; check bid/ask vs mid consistency")

    n_bid = df["fill_bid"].astype(int).to_numpy(dtype=float)
    n_ask = df["fill_ask"].astype(int).to_numpy(dtype=float)

    delta = np.concatenate([delta_bid, delta_ask])
    n = np.concatenate([n_bid, n_ask])

    return delta, n
