import numpy as np
import pytest
import torch
from scipy.stats import levy_stable

from estimators import (
    ALPHA_MS,
    ALPHA_WINDOW,
    alpha_estimator,
    estimate_alpha,
    estimate_llc,
    increment_diagnostics,
    true_dln_learning_coefficient,
)
from models import DLN


def test_true_lambda_of_regular_model_is_half_the_parameter_count():
    assert true_dln_learning_coefficient(5, [5, 5]) == pytest.approx(25 / 2)


def test_true_lambda_matches_reduced_rank_regression():
    # Aoyagi & Watanabe 2005, M + H + N + r even: (2(H+r)(M+N) - (M-N)^2 - (H+r)^2) / 8
    M, H, N, r = 4, 4, 4, 2
    expected = (2 * (H + r) * (M + N) - (M - N) ** 2 - (H + r) ** 2) / 8
    assert true_dln_learning_coefficient(r, [M, H, N]) == pytest.approx(expected)


@pytest.mark.parametrize("alpha", [1.2, 1.5, 2.0])
def test_alpha_estimator_recovers_tail_index_of_iid_stable_samples(alpha):
    X = levy_stable.rvs(alpha, 0.0, size=(36 * 36 * 100, 1), random_state=0)
    X = torch.from_numpy(X).float()
    assert alpha_estimator(36, X).item() == pytest.approx(alpha, abs=0.15)


@pytest.mark.parametrize("alpha", [1.5, 2.0])
def test_estimate_alpha_recovers_tail_index_of_a_levy_motion(alpha):
    # The trajectory HD models: positions are cumulative sums of i.i.d. stable increments
    assert all(ALPHA_WINDOW % m == 0 for m in ALPHA_MS)
    generator = torch.Generator().manual_seed(0)
    steps = levy_stable.rvs(alpha, 0.0, size=(ALPHA_WINDOW + 1, 50), random_state=0)
    iterates = torch.from_numpy(steps).float().cumsum(0)
    median, maximum = estimate_alpha(iterates, generator=generator)
    assert median == pytest.approx(alpha, abs=0.4)
    assert maximum >= median


def test_increment_diagnostics_flags_frozen_and_mean_reverting_paths():
    generator = torch.Generator().manual_seed(0)
    frozen = torch.ones(ALPHA_WINDOW + 1, 10)
    frozen[::50] += 1.0
    assert increment_diagnostics(frozen, generator=generator)[1] > 0.9

    # i.i.d. positions around a fixed point: increments are anticorrelated at about -0.5
    around_a_minimum = torch.randn(ALPHA_WINDOW + 1, 10, generator=generator)
    lag1, zero_fraction = increment_diagnostics(around_a_minimum, generator=generator)
    assert lag1 == pytest.approx(-0.5, abs=0.15)
    assert zero_fraction == 0


def test_estimate_llc_leaves_the_model_untouched():
    torch.manual_seed(0)
    model = DLN([3, 4, 3], sigma=0.5)
    before = [p.detach().clone() for p in model.parameters()]
    lambdahat = estimate_llc(model, torch.randn(1000, 3), num_steps=20)
    assert np.isfinite(lambdahat)
    for p, p_before in zip(model.parameters(), before):
        assert torch.equal(p, p_before)

