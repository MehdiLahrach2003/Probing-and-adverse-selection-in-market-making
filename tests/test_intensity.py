import math
import pytest

from optimal_quoting.model.intensity import intensity_exp


def test_intensity_exp_basic():
    assert math.isclose(intensity_exp(2.0, 1.0, 0.0), 2.0)

def test_intensity_exp_monotone():
    lam0 = intensity_exp(2.0, 1.0, 0.0)
    lam1 = intensity_exp(2.0, 1.0, 1.0)
    assert lam1 < lam0

@pytest.mark.parametrize("A,k,delta", [(0.0,1.0,0.0), (1.0,0.0,0.0), (1.0,1.0,-0.1)])
def test_intensity_exp_invalid(A,k,delta):
    with pytest.raises(ValueError):
        intensity_exp(A,k,delta)
