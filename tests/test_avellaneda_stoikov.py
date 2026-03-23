import pytest

from optimal_quoting.model.avellaneda_stoikov import ASParams, as_deltas


def test_as_symmetry_at_zero_inventory():
    p = ASParams(gamma=0.1, sigma=0.02, k=1.0, T=100.0)
    db, da = as_deltas(q=0.0, t=0.0, p=p)
    assert db == pytest.approx(1.0)
    assert da == pytest.approx(1.0)


def test_as_inventory_skew_direction():
    p = ASParams(gamma=0.1, sigma=0.02, k=2.0, T=100.0)

    db0, da0 = as_deltas(q=0.0, t=0.0, p=p)
    dbp, dap = as_deltas(q=+10.0, t=0.0, p=p)
    dbn, dan = as_deltas(q=-10.0, t=0.0, p=p)

    # If q>0 (long), ask should widen, bid should tighten
    assert dap > da0
    assert dbp < db0

    # If q<0 (short), ask should tighten, bid should widen
    assert dan < da0
    assert dbn > db0
