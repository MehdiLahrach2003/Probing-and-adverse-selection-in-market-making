import numpy as np

from optimal_quoting.strategy.probing import ProbingConfig, compute_probing_quotes


def test_probing_widens_deltas_when_enabled():
    rng = np.random.default_rng(0)
    cfg = ProbingConfig(p_explore=1.0, jitter=0.5, widen_only=True)

    q = 0.0
    mid = 100.0
    base_spread = 0.2
    phi = 0.0

    quotes = compute_probing_quotes(mid, q, base_spread, phi, cfg, rng)
    assert quotes.delta_bid >= 0.1
    assert quotes.delta_ask >= 0.1
