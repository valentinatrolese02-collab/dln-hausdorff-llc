import itertools
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import levy_stable, rankdata, spearmanr
from torchvision import transforms

import analyse
from cnn.alpha import increment_series, iterate_series, load_tail_index, tail_index_summary
from cnn.calibrate_sgld import assess, pick
from cnn.data import CIFAR_MEAN, CIFAR_STD, normalise
from cnn.llc import equivalent_epochs, estimate_llc
from cnn.train_vgg import ProjectionRecorder, l2_penalty
from estimators import ALPHA_MS, alpha_estimator, alpha_window, projected_alpha, random_directions
from models import VGG

ROOT = Path(__file__).resolve().parents[1]


def hd_estimator_vector_projected(iterate_matrix, directions, sz, ms):
    """Port of estimator_vector_projected in HD's alpha_estimator_vgg.py, with the directions passed in."""
    iterate_matrix_zm = iterate_matrix - torch.mean(iterate_matrix, axis=0).view(1, -1)
    proj_alpha = []
    for rand_direction in directions:
        projected = torch.mm(iterate_matrix_zm, rand_direction.view(-1, 1))
        cur_alpha_est = [alpha_estimator(mm, projected[-1 * sz :, :]).item() for mm in ms]
        proj_alpha.append(np.median(cur_alpha_est))
    return np.median(proj_alpha), np.max(proj_alpha)


@pytest.mark.parametrize("alpha", [2.0, 1.5, 1.0])
def test_projected_alpha_recovers_tail_index_of_iid_stable_samples(alpha):
    samples = levy_stable.rvs(alpha, 0.0, size=(36 * 1000, 4), random_state=0)
    estimates = projected_alpha(torch.from_numpy(samples))
    assert np.median(estimates) == pytest.approx(alpha, abs=0.15)


@pytest.mark.parametrize("steps, window", [(1562, 1548), (781, 756), (390, 360), (195, 180), (62, 36)])
def test_window_is_the_largest_multiple_of_36_that_fits(steps, window):
    assert alpha_window(steps) == window
    assert all(window % m == 0 for m in ALPHA_MS)


def test_window_refuses_a_final_epoch_that_is_too_short():
    with pytest.raises(ValueError):
        alpha_window(36)


def synthetic_recording():
    """A trajectory written through a real model, stored both in full and by ProjectionRecorder."""
    torch.manual_seed(0)
    model = nn.Linear(6, 3)
    recorder = ProjectionRecorder(model, torch.Generator().manual_seed(1))
    recorder.start()
    weight_steps = torch.from_numpy(levy_stable.rvs(1.5, 0.0, size=(100, 18), random_state=0)).float() * 1e-3
    full = []
    for step in weight_steps:
        with torch.no_grad():
            model.weight += step.view(3, 6)
            model.bias += 1.0
        recorder.record()
        full.append(model.weight.detach().flatten().double().clone())
    return recorder, torch.stack(full)


def test_stored_projections_match_hd_estimator_on_the_full_iterates():
    recorder, full = synthetic_recording()
    assert list(recorder.projections) == ["weight"]  # the bias is skipped, as in HD
    N = alpha_window(len(full))
    directions = recorder.directions["weight"]

    stored = torch.from_numpy(np.stack(recorder.projections["weight"]))
    from_stored = projected_alpha(iterate_series(stored, N))
    median, maximum = hd_estimator_vector_projected(full, directions, N, ALPHA_MS)
    assert np.median(from_stored) == pytest.approx(median, rel=1e-9)
    assert np.max(from_stored) == pytest.approx(maximum, rel=1e-9)

    increments = full[-(N + 1) :].diff(dim=0)
    from_full = projected_alpha((increments - increments.mean(dim=0, keepdim=True)) @ directions.T)
    assert projected_alpha(increment_series(stored, N)) == pytest.approx(from_full, rel=1e-9)


def test_recorder_leaves_the_training_rng_alone():
    model = nn.Linear(6, 3)
    before = torch.get_rng_state()
    recorder = ProjectionRecorder(model, torch.Generator().manual_seed(1))
    recorder.start()
    recorder.record()
    assert torch.equal(before, torch.get_rng_state())
    assert recorder.zero_fraction == [[1.0]]


def test_random_directions_have_unit_norm():
    directions = random_directions(50, 10, torch.Generator().manual_seed(0), dtype=torch.float64)
    assert directions.norm(dim=1) == pytest.approx(torch.ones(10, dtype=torch.float64))


def test_flagged_runs():
    generator = torch.Generator().manual_seed(0)
    # A random walk has independent increments; i.i.d. positions have lag-1 about -0.5
    walk = torch.randn(361, 10, generator=generator, dtype=torch.float64).cumsum(0)
    around_a_minimum = torch.randn(361, 10, generator=generator, dtype=torch.float64)
    no_zeros = np.zeros((361, 1))
    assert not tail_index_summary({"w": walk}, no_zeros)["flagged"]
    assert tail_index_summary({"w": around_a_minimum}, no_zeros)["flagged"]
    assert tail_index_summary({"w": walk}, np.ones((361, 1)))["flagged"]
    assert tail_index_summary({"w": torch.ones(361, 10, dtype=torch.float64)}, no_zeros)["flagged"]


def test_vgg_matches_hd_construction():
    vgg4 = VGG("VGG4")
    assert sum(p.numel() for p in vgg4.parameters()) == 3 * 512 * 9 + 512 + 131072 * 10 + 10
    assert [name for name, p in vgg4.named_parameters() if p.ndim >= 2] == ["features.0.weight", "classifier.weight"]
    assert not any(isinstance(m, (nn.BatchNorm2d, nn.Dropout)) for m in vgg4.modules())
    for arch in ("VGG4", "VGG6"):
        assert VGG(arch)(torch.zeros(2, 3, 32, 32)).shape == (2, 10)


def test_normalise_matches_hd_transform():
    image = np.random.default_rng(0).integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    hd = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    assert torch.allclose(normalise(image[None])[0], hd(image))


def test_l2_penalty_is_the_sum_of_squared_parameters():
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))
        model.bias.copy_(torch.tensor([3.0]))
    assert l2_penalty(model).item() == pytest.approx(14.0)


class BinaryLogistic(nn.Module):
    """Logistic regression with p parameters, written as two-class logits for cross-entropy."""

    def __init__(self, p):
        super().__init__()
        self.linear = nn.Linear(p, 1, bias=False)

    def forward(self, x):
        logit = self.linear(x)
        return torch.cat([torch.zeros_like(logit), logit], dim=1)


def fit_logistic(n=2000, p=5):
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(n, p, generator=generator)
    teacher = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.25])
    y = (torch.rand(n, generator=generator) < torch.sigmoid(x @ teacher)).long()
    torch.manual_seed(0)
    model = BinaryLogistic(p)
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=200, tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return model, x, y


LOGISTIC_SGLD = {"lr": 2e-3, "localization": 1.0, "num_steps": 4000, "burn_in": 0.9, "num_chains": 2, "batch_size": 512}


def test_llc_of_a_regular_model_is_half_the_parameter_count():
    # Over 6 sampler seeds these settings gave a mean of 2.50 and a largest single error of 0.43
    model, x, y = fit_logistic()
    result = estimate_llc(model, x, y, LOGISTIC_SGLD, seed=0, device="cpu")
    assert result["nbeta"] == pytest.approx(2000 / math.log(2000))
    assert result["llc_mean"] == pytest.approx(2.5, abs=0.5)


def test_llc_is_reproducible_and_keeps_rng_and_weights():
    model, x, y = fit_logistic()
    sgld = LOGISTIC_SGLD | {"num_steps": 100}
    before_weights = model.linear.weight.detach().clone()
    torch.manual_seed(7)
    before_rng = torch.get_rng_state()
    first = estimate_llc(model, x, y, sgld, seed=3, device="cpu")
    assert torch.equal(before_rng, torch.get_rng_state())
    assert torch.equal(before_weights, model.linear.weight)

    assert estimate_llc(model, x, y, sgld, seed=3, device="cpu")["llc_chains"] == first["llc_chains"]
    assert estimate_llc(model, x, y, sgld, seed=4, device="cpu")["llc_chains"] != first["llc_chains"]
    assert first["llc_chains"][0] != first["llc_chains"][1]


def test_calibration_trace_gives_the_same_llc_as_the_sweep():
    model, x, y = fit_logistic()
    sgld = LOGISTIC_SGLD | {"num_steps": 100}
    sweep = estimate_llc(model, x, y, sgld, seed=3, device="cpu")
    full = estimate_llc(model, x, y, sgld, seed=3, device="cpu", keep_burnin=True)
    assert full["llc_chains"] == pytest.approx(sweep["llc_chains"])
    assert full["loss_trace"].shape == (2, 100)
    assert sweep["loss_trace"].shape == (2, 10)


def calibration_result(trace, init_loss=1.0, nbeta=100.0, num_kept=10, diverged=False):
    trace = np.array(trace, dtype=float)
    llc = nbeta * (trace[:, -num_kept:].mean(axis=1) - init_loss)
    return {"loss_trace": trace, "num_kept": num_kept, "init_loss": init_loss, "nbeta": nbeta, "llc_chains": llc.tolist(), "diverged": diverged}


@pytest.mark.parametrize(
    "trace, reason",
    [
        ([[1.5] * 100, [1.5] * 100], "stable"),
        ([[0.5] * 100, [0.5] * 100], "negative LLC"),
        ([[3.0] * 100, [3.0] * 100], "left the basin"),
        ([list(np.linspace(1.0, 2.0, 100))] * 2, "no plateau"),
    ],
)
def test_calibration_rule(trace, reason):
    stable, found = assess(calibration_result(trace))
    assert found.startswith(reason)
    assert stable == (reason == "stable")


def test_calibration_picks_the_smallest_localisation_then_step():
    entries = [
        {"lr": 1e-6, "localization": 10.0, "stable": True},
        {"lr": 3e-6, "localization": 1.0, "stable": True},
        {"lr": 1e-6, "localization": 1.0, "stable": True},
        {"lr": 1e-7, "localization": 1.0, "stable": False},
    ]
    assert pick(entries) == entries[2]
    assert pick([entries[3]]) is None


def test_compute_cost_counts_all_chains():
    assert equivalent_epochs({"num_chains": 2, "num_steps": 3000, "batch_size": 512}, 50000) == pytest.approx(61.44)


def enumerated_spearman_p(x, y):
    """Two-sided permutation p by listing every pairing, in exact integers."""
    a, b = (2 * rankdata(x)).astype(int), (2 * rankdata(y)).astype(int)
    n = len(a)
    centre = int(a.sum()) * int(b.sum())
    observed = abs(n * int(a @ b) - centre)
    return sum(abs(n * int(a @ b[list(perm)]) - centre) >= observed for perm in itertools.permutations(range(n))) / math.factorial(n)


@pytest.mark.parametrize(
    "x, y",
    [
        ([1, 2, 3, 4, 5, 6, 7, 8], [2, 1, 4, 3, 6, 5, 8, 7]),  # no ties
        ([1, 2, 3, 4, 5, 6, 7, 8], [5, 1, 8, 2, 7, 3, 6, 4]),  # no ties, weak
        ([1, 1, 1, 2, 2, 2, 3, 3], [0.3, 0.1, 0.2, 0.6, 0.4, 0.9, 0.7, 0.8]),  # ties in x, as in a factor
        ([1, 1, 2, 2, 3, 3, 4, 4], [1, 1, 1, 2, 2, 3, 3, 3]),  # half-integer midranks in both
    ],
)
def test_exact_spearman_p_matches_enumeration(x, y):
    assert analyse.exact_spearman_p(np.array(x), np.array(y)) == pytest.approx(enumerated_spearman_p(x, y), rel=1e-12)


def test_pairing_counts_cover_all_permutations():
    assert analyse.pairing_counts(tuple(range(1, 15)), tuple(range(1, 15)))[1].sum() == math.factorial(14)
    # n = 20 with x on 4 levels, the largest within-factor shape, stays exact and is quick
    levels = tuple(sorted(analyse.integer_ranks(np.repeat([1, 2, 3, 4], 5)).tolist()))
    assert int(analyse.pairing_counts(tuple(range(1, 21)), levels)[1].sum()) == math.factorial(20)


def test_exact_spearman_p_at_perfect_order_is_two_pairings():
    assert analyse.exact_spearman_p(np.arange(10.0), np.arange(10.0)) == pytest.approx(2 / math.factorial(10))


def test_spearman_method_depends_on_n():
    rng = np.random.default_rng(0)
    assert np.isnan(analyse.spearman(np.arange(7.0), rng.normal(size=7))[1])
    assert np.isfinite(analyse.spearman(np.arange(8.0), rng.normal(size=8))[1])

    for n, is_exact in [(20, True), (21, False)]:
        x = np.repeat(np.arange(4.0), n // 4 + 1)[:n]
        y = x + rng.normal(size=n)
        t_p = spearmanr(x, y).pvalue
        assert (analyse.spearman(x, y)[1] != t_p) == is_exact


def test_too_few_runs_gets_rho_but_no_p_or_verdict():
    assert analyse.fmt_p(float("nan"), 7) == "too few runs"
    text = analyse.describe(*analyse.spearman(np.arange(7.0), np.arange(7.0)))
    assert "too few runs" in text and "rho = 1" in text and "p " not in text.replace("relationship", "")
    assert "monotonic" not in text


def test_p_is_never_printed_as_zero():
    assert analyse.fmt_p(0.0, 10) == f"< {1 / math.factorial(10):.3g}"
    assert analyse.fmt_p(0.0, 30).startswith("< ")
    # scipy's t tail really is 0 at perfect order, so this path is reachable
    rho, p, n = analyse.spearman(np.arange(30.0), np.arange(30.0))
    assert "p < " in analyse.describe(rho, p, n)


def test_trend_verdict_needs_eight_runs():
    baseline = {"lr": 0.01, "batch_size": 128, "momentum": 0.0, "l2": 0.0}

    def report(n):
        strength = np.linspace(0.001, 0.01, n)
        df = pd.DataFrame({**baseline, "lr": strength}).assign(**{m: -strength for m in analyse.PREDICTED})
        return analyse.trend_section(df, baseline, ["lr"])

    assert "as predicted" not in report(7) and "too few runs" in report(7)
    # Every metric falls with strength, and only test accuracy was predicted to rise
    assert "too few runs" not in report(8) and "5 as predicted, 1 opposite to prediction" in report(8)


@pytest.mark.skipif(not (ROOT / "data/cifar-10-batches-py").exists(), reason="CIFAR-10 not downloaded")
def test_smoke_run_end_to_end(tmp_path):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_sweep.py"), "--smoke", "--out-dir", str(tmp_path), "--device", "cpu", "--data-root", str(ROOT / "data")],
        check=True,
        cwd=ROOT,
        timeout=600,
    )
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert len(summary) == 3
    assert summary[["train_acc", "test_acc", "beta_iter_max", "beta_incr_max", "llc_mean"]].notna().all(axis=None)
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "figures/scatter.png").exists()
    run_dir = next((tmp_path / "runs").iterdir())
    assert {"state_dict.pt", "projections.npz", "train.json", "run.json"} <= {p.name for p in run_dir.iterdir()}
    assert load_tail_index(run_dir / "projections.npz")["window"] == 36

    # Resuming skips finished runs and leaves the summary unchanged
    resumed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_sweep.py"), "--smoke", "--out-dir", str(tmp_path), "--device", "cpu", "--data-root", str(ROOT / "data")],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert resumed.stdout.count("skip ") == 3
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "summary.csv"), summary)
