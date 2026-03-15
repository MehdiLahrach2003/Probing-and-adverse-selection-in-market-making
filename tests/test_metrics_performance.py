import numpy as np
import pandas as pd

from optimal_quoting.metrics.performance import performance_summary


def test_performance_summary_smoke():
    df = pd.DataFrame(
        {
            "equity": np.cumsum(np.ones(100)),
            "inventory": np.zeros(100),
        }
    )
    out = performance_summary(df)
    assert "pnl_final" in out
    assert "sharpe" in out
