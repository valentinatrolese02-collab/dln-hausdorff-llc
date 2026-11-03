import json
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
from scipy.stats import levy_stable
from torchvision import transforms

import analyse
import run_one as run_one_script
import run_sweep
from cnn.alpha import ALPHA_MS, ProjectionBuffer, iterate_series, increment_series, projected_alpha, random_directions, tail_index
from cnn.calibrate_sgld import assess
from cnn.data import CIFAR_MEAN, CIFAR_STD, normalise
from cnn.llc import equivalent_epochs, estimate_llc, llc_flag
from cnn.sweep import distinct_runs, load_sweep, load_timeseries, run_config, run_id, strength_order
from cnn.train_vgg import TIMESERIES_COLUMNS, checkpoint_steps, logged_steps, minibatch_indices, train
from models import VGG
from utils import alpha_estimator

ROOT = Path(__file__).resolve().parents[1]


def hd_estimator_vector_projected(iterate_matrix, directions, sz, ms):
    """Port of estimator_vector_projected in HD's alpha_estimator_vgg.py, with the directions passed in."""
    iterate_matrix_zm = iterate_matrix - torch.mean(iterate_matrix, axis=0).view(1, -1)
    proj_alpha = []
    for rand_direction in directions:
        projected = torch.mm(iterate_matrix_zm, rand_direction.view(-1, 1))
        cur_alpha_est = [alpha_estimator(mm, projected[-1 * sz :, :]).item() for mm in ms]
        proj_alpha.append(np.median(cur_alpha_est))
    return np.median(proj_alpha)


def trajectory_through_buffer(window, steps=500):
    """A heavy-tailed walk written through a real model, stored both in full and by the rolling buffer."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 3), nn.Linear(3, 4))
    buffer = ProjectionBuffer(model, torch.Generator().manual_seed(1), window)
    weights = [m.weight for m in model]
    moves = [torch.from_numpy(levy_stable.rvs(a, 0.0, size=(steps, w.numel()), random_state=i)).float() * 1e-3 for i, (a, w) in enumerate(zip((1.5, 1.8), weights))]
    full = [[], []]
    for t in range(steps):
        with torch.no_grad():
            for i, w in enumerate(weights):
                w += moves[i][t].view_as(w)
            model[0].bias += 1.0  # biases move too, and must not be stored
        buffer.push()
        for i, w in enumerate(weights):
            full[i].append(w.detach().flatten().double().clone())
    return buffer, [torch.stack(rows) for rows in full]


@pytest.mark.parametrize("window", [361, 181])
def test_rolling_buffer_alpha_equals_alpha_from_full_iterates(window):
    buffer, full = trajectory_through_buffer(window)
    assert list(buffer.iterates) == ["0.weight", "1.weight"]
    assert buffer.is_full()
    summary = tail_index(*buffer.series())

    last = [rows[-window:] for rows in full]
    directions = list(buffer.directions.values())
    alpha_iter = [hd_estimator_vector_projected(rows, d, window - 1, ALPHA_MS) for rows, d in zip(last, directions)]
    alpha_incr = []
    for rows, d in zip(last, directions):
        increments = rows.diff(dim=0)
        alpha_incr.append(np.median(projected_alpha((increments - increments.mean(dim=0, keepdim=True)) @ d.T)))
    for name, values in [("iter", alpha_iter), ("incr", alpha_incr)]:
        assert summary[f"beta_{name}_max"] == pytest.approx(max(values), rel=1e-9)
        assert summary[f"beta_{name}_median"] == pytest.approx(np.median(values), rel=1e-9)


def test_buffer_needs_a_full_window_and_stays_at_its_length():
    model = nn.Linear(2, 2)
    buffer = ProjectionBuffer(model, torch.Generator().manual_seed(0), window=5)
    for pushed in range(1, 8):
        with torch.no_grad():
            model.weight += 1.0
        buffer.push()
        assert buffer.is_full() == (pushed >= 5)
    series, zero_fraction = buffer.series()
    assert series["weight"].shape == (5, 10)
    assert zero_fraction.shape == (4, 1)


def test_buffer_counts_exact_zero_updates_and_leaves_the_training_rng_alone():
    model = nn.Linear(6, 3)
    before = torch.get_rng_state()
    buffer = ProjectionBuffer(model, torch.Generator().manual_seed(1), window=4)
    buffer.push()
    with torch.no_grad():
        model.weight[0, :3] += 1.0  # 3 of the 18 weights move
    buffer.push()
    assert list(buffer.zero_fraction) == [[1.0], [15 / 18]]
    assert torch.equal(before, torch.get_rng_state())


def test_random_directions_have_unit_norm():
    directions = random_directions(50, 10, torch.Generator().manual_seed(0))
    assert directions.dtype == torch.float64
    assert directions.norm(dim=1) == pytest.approx(torch.ones(10, dtype=torch.float64))


@pytest.mark.parametrize("alpha", [2.0, 1.5, 1.0])
def test_projected_alpha_recovers_tail_index_of_iid_stable_samples(alpha):
    samples = levy_stable.rvs(alpha, 0.0, size=(36 * 1000, 4), random_state=0)
    assert np.median(projected_alpha(torch.from_numpy(samples))) == pytest.approx(alpha, abs=0.15)


def test_flags_and_variants():
    generator = torch.Generator().manual_seed(0)
    # A random walk has independent increments; i.i.d. positions have lag-1 about -0.5
    walk = torch.randn(361, 10, generator=generator, dtype=torch.float64).cumsum(0)
    around_a_minimum = torch.randn(361, 10, generator=generator, dtype=torch.float64)
    no_zeros = np.zeros((360, 1))
    assert not tail_index({"w": walk}, no_zeros)["alpha_flag"]
    near_minimum = tail_index({"w": around_a_minimum}, no_zeros)
    assert near_minimum["alpha_flag"]
    assert near_minimum["lag1"] == pytest.approx(-0.5, abs=0.1)
    # What the flag exists for: block sums telescope, so alpha leaves (0, 2], past infinity to negative values at this extreme
    assert not 0 < near_minimum["beta_incr_max"] <= 2
    assert tail_index({"w": walk}, np.ones((360, 1)))["alpha_flag"]
    assert tail_index({"w": torch.ones(361, 10, dtype=torch.float64)}, no_zeros)["alpha_flag"]  # NaN lag-1 flags too
    assert tail_index({"w": walk}, no_zeros)["beta_iter_max"] == pytest.approx(1.0, abs=0.4)  # slowly moving path reads about 1
    assert iterate_series(walk).shape == increment_series(walk).shape == (360, 10)


def test_checkpoint_scheduler():
    assert checkpoint_steps(15000, 12) == list(range(1250, 15001, 1250))
    assert checkpoint_steps(300, 3) == [100, 200, 300]
    assert checkpoint_steps(1000, 7)[-1] == 1000
    assert len(set(checkpoint_steps(1000, 7))) == 7
    full = logged_steps(15000, 250, checkpoint_steps(15000, 12))
    assert full == list(range(250, 15001, 250))
    assert logged_steps(10, 4, [3, 10]) == [3, 4, 8, 10]


def test_minibatches_drop_the_last_batch_and_reshuffle():
    generator = torch.Generator().manual_seed(0)
    batches = minibatch_indices(10, 4, generator)
    first_epoch = [next(batches) for _ in range(2)]
    second_epoch = [next(batches) for _ in range(2)]
    assert all(len(b) == 4 for b in first_epoch + second_epoch)
    assert len(set(torch.cat(first_epoch).tolist())) == 8
    assert not torch.equal(torch.cat(first_epoch), torch.cat(second_epoch))


def test_config_is_the_experiment_of_the_brief():
    sweep = load_sweep(ROOT / "configs/sweep.yaml")
    assert list(sweep["factors"]) == ["lr", "batch_size", "momentum"]
    assert len(distinct_runs(sweep)) == 27  # 9 settings, sharing the baseline, x 3 seeds
    assert all((sweep["window"] - 1) % m == 0 for m in sweep["ms"])
    smoke = load_sweep(ROOT / "configs/sweep.yaml", smoke=True)
    assert (smoke["subset"], smoke["steps"], smoke["num_checkpoints"], smoke["window"], smoke["seeds"]) == (2000, 300, 3, 181, [0])
    assert all(len(values) == 2 for values in smoke["factors"].values())


def test_baseline_is_one_run_whichever_sweep_names_it():
    sweep = load_sweep(ROOT / "configs/sweep.yaml")
    ids = {run_id(sweep, run_config(sweep, factor, value), 0) for factor, value in [("lr", 0.01), ("batch_size", 128), ("momentum", 0)]}
    assert ids == {"VGG4_lr0.01_bs128_m0_seed0"}
    with pytest.raises(ValueError):
        run_config(sweep, "lr", 0.5)


def test_strength_order():
    assert strength_order("lr", [0.01, 0.001]) == [0.001, 0.01]
    assert strength_order("batch_size", [32, 512]) == [512, 32]
    assert strength_order("momentum", [0.9, 0.0]) == [0.0, 0.9]


def test_dry_run_prints_runs_steps_and_sgld_cost_only(tmp_path):
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_sweep.py"), "--dry-run", "--sgld", str(tmp_path / "absent.yaml")],
        check=True, cwd=ROOT, capture_output=True, text=True,
    ).stdout.splitlines()  # fmt: skip
    assert len(out) == 3
    assert out[0] == "runs: 27"
    assert out[1] == "SGD steps: 405000"
    assert out[2].startswith(f"SGLD: {27 * 12 * equivalent_epochs({'num_chains': 2, 'num_steps': 1000, 'batch_size': 512}, 50000):.0f} equivalent")


def test_list_gives_one_line_per_run():
    lines = subprocess.run([sys.executable, str(ROOT / "scripts/run_sweep.py"), "--list"], check=True, cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    assert len(lines) == 27
    assert lines[0] == "--factor lr --value 0.001 --seed 0"


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
    model, x, y = fit_logistic()
    result = estimate_llc(model, x, y, LOGISTIC_SGLD, seed=[1, 2], device="cpu")
    assert result["nbeta"] == pytest.approx(2000 / math.log(2000))
    assert result["llc/mean"] == pytest.approx(2.5, abs=0.5)
    assert llc_flag(result) == "ok"


def test_llc_is_reproducible_and_keeps_rng_and_weights():
    model, x, y = fit_logistic()
    sgld = LOGISTIC_SGLD | {"num_steps": 100}
    before_weights = model.linear.weight.detach().clone()
    torch.manual_seed(7)
    before_rng = torch.get_rng_state()
    first = estimate_llc(model, x, y, sgld, seed=[3, 4], device="cpu")
    assert torch.equal(before_rng, torch.get_rng_state())
    assert torch.equal(before_weights, model.linear.weight)
    assert estimate_llc(model, x, y, sgld, seed=[3, 4], device="cpu")["llc/mean"] == first["llc/mean"]
    assert estimate_llc(model, x, y, sgld, seed=[5, 6], device="cpu")["llc/mean"] != first["llc/mean"]


@pytest.mark.parametrize(
    "res, flag",
    [
        ({"llc/mean": float("nan"), "diverged": True}, "diverged"),
        ({"llc/mean": float("inf"), "loss/trace": np.ones((2, 4))}, "diverged"),
        ({"llc/mean": -5.0, "loss/trace": np.full((2, 4), 3.0)}, "negative"),
        ({"llc/mean": 5.0, "loss/trace": np.array([[0.5] * 4, [2.5] * 4])}, "left_basin"),
        ({"llc/mean": 5.0, "loss/trace": np.full((2, 4), 0.5)}, "ok"),
    ],
)
def test_llc_flag(res, flag):
    assert llc_flag(res) == flag


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
    res = {"loss/trace": np.array(trace, dtype=float), "init_loss": 1.0, "nbeta": 100.0}
    stable, found, _ = assess(res, num_steps=100, burn_in=0.9)
    assert found.startswith(reason)
    assert stable == (reason == "stable")


def fake_cifar(root="data", subset=None):
    generator = torch.Generator().manual_seed(0)
    return torch.randn(64, 3, 32, 32, generator=generator), torch.randint(0, 10, (64,), generator=generator), torch.randn(32, 3, 32, 32, generator=generator), torch.randint(0, 10, (32,), generator=generator)


@pytest.fixture
def tiny_sweep(monkeypatch):
    monkeypatch.setattr("cnn.train_vgg.load_cifar10", fake_cifar)
    monkeypatch.setattr(run_one_script, "load_cifar10", fake_cifar)
    return {
        "arch": "VGG4", "steps": 30, "seeds": [0], "log_every": 10, "num_checkpoints": 2, "window": 19, "ms": [2, 3, 6, 9], "num_projections": 3, "subset": None,
        "baseline": {"lr": 0.001, "batch_size": 8, "momentum": 0.0}, "factors": {"lr": [0.001], "momentum": [0.0, 0.5]},
        "sgld": {"lr": 1e-7, "localization": 1.0, "num_steps": 10, "burn_in": 0.5, "num_chains": 2, "batch_size": 8},
    }  # fmt: skip


def train_tiny(sweep, out, **overrides):
    settings = {k: sweep[k] for k in ("steps", "log_every", "num_checkpoints", "window", "ms", "num_projections", "subset")} | overrides
    return train(sweep["arch"], **sweep["baseline"], seed=0, run_dir=out, csv_path=out / "series.csv", device="cpu", **settings)


def test_train_writes_the_columns_and_is_not_changed_by_measuring(tiny_sweep, tmp_path):
    metrics = train_tiny(tiny_sweep, tmp_path / "a")
    df = pd.read_csv(tmp_path / "a/series.csv")
    assert list(df.columns) == TIMESERIES_COLUMNS
    assert df["step"].tolist() == [10, 15, 20, 30]
    assert metrics["checkpoints"] == [15, 30]
    assert metrics["epochs"] == pytest.approx(30 * 8 / 64)
    assert df.loc[df["step"].isin([10, 20]), "beta_incr_max"].isna().all()  # not a checkpoint
    assert df.loc[df["step"] == 15, "beta_incr_max"].isna().all()  # window of 19 not full yet
    assert df.loc[df["step"] == 30, ["beta_iter_max", "beta_incr_max", "lag1", "zero_fraction", "alpha_flag"]].notna().all(axis=None)
    assert df["llc_mean"].isna().all() and df["llc_flag"].isna().all()
    assert {p.name for p in (tmp_path / "a/weights").iterdir()} == {"step_15.pt", "step_30.pt"}

    # Another window and other checkpoints change what is measured, never the training run
    train_tiny(tiny_sweep, tmp_path / "b", window=10, ms=[3, 9], num_checkpoints=3, log_every=7)
    first, second = (torch.load(tmp_path / d / "state_dict.pt") for d in "ab")
    assert all(torch.equal(first[k], second[k]) for k in first)


def test_run_one_trains_once_and_resumes_the_llc(tiny_sweep, tmp_path, capsys):
    sgld = tiny_sweep["sgld"]
    run_one_script.run_one(tiny_sweep, "lr", 0.001, 0, None, tmp_path, "cpu", train_only=True)
    csv = tmp_path / "timeseries/VGG4_lr0.001_bs8_m0_seed0.csv"
    assert pd.read_csv(csv)["llc_mean"].isna().all()

    run_one_script.run_one(tiny_sweep, "momentum", 0.0, 0, sgld, tmp_path, "cpu")  # the same baseline run, by its other name
    done = pd.read_csv(csv, dtype={"llc_flag": "str"})
    assert "training finished, skip" in capsys.readouterr().out
    assert done.loc[done["step"].isin([15, 30]), "llc_flag"].notna().all()

    # Lose one estimate, as a crash would: only that one runs again, with the same chain seeds
    lost = done.copy()
    lost.loc[lost["step"] == 30, ["llc_mean", "llc_std", "llc_flag"]] = [float("nan"), float("nan"), float("nan")]
    lost.to_csv(csv, index=False, na_rep="NaN")
    run_one_script.run_one(tiny_sweep, "lr", 0.001, 0, sgld, tmp_path, "cpu")
    out = capsys.readouterr().out
    assert "step 15: LLC" not in out and "step 30: LLC" in out
    pd.testing.assert_frame_equal(pd.read_csv(csv, dtype={"llc_flag": "str"}), done)

    with pytest.raises(SystemExit, match="SGLD"):
        run_one_script.run_one(tiny_sweep, "lr", 0.001, 0, sgld | {"lr": 1e-6}, tmp_path, "cpu")
    with pytest.raises(SystemExit, match="settings"):
        run_one_script.run_one(tiny_sweep | {"steps": 40}, "lr", 0.001, 0, sgld, tmp_path, "cpu")


def fake_results(root):
    """Two seeds at each value, with every kind of flagged point, and one run with no results yet."""
    sweep = {
        "arch": "VGG4", "steps": 100, "seeds": [0, 1], "log_every": 50, "num_checkpoints": 2, "window": 37, "ms": [2, 3, 6], "num_projections": 3, "subset": None,
        "baseline": {"lr": 0.01, "batch_size": 128, "momentum": 0.0},
        "factors": {"lr": [0.001, 0.01], "batch_size": [128, 512], "momentum": [0.0, 0.9]},
    }  # fmt: skip
    run_one_script.write_resolved(root, sweep, {"lr": 1e-6, "localization": 1.0, "num_steps": 100, "burn_in": 0.9, "num_chains": 2, "batch_size": 512})
    (root / "timeseries").mkdir()
    for factor, value, seed in distinct_runs(sweep):
        rid = run_id(sweep, run_config(sweep, factor, value), seed)
        if rid == "VGG4_lr0.01_bs512_m0_seed1":
            continue
        shift = seed * 0.01 + (value if factor == "lr" else 0.0)
        pd.DataFrame(
            {
                "step": [50, 100], "train_loss": [1.0 + shift, 0.1 + shift], "test_loss": [1.2, 0.9], "test_acc": [0.5 + shift, 0.6 + shift],
                "llc_mean": [-20.0, 300.0 + 100 * shift], "llc_std": [5.0, 7.0], "llc_flag": ["negative" if seed == 0 else "diverged", "ok"],
                "beta_iter_max": [1.0, 1.0 + shift], "beta_iter_median": [1.0, 1.0], "beta_incr_max": [2.5, 1.8 + shift], "beta_incr_median": [2.2, 1.7],
                "lag1": [-0.3, -0.05], "zero_fraction": [0.0, 0.0], "alpha_flag": [1.0, 0.0],
            }
        ).to_csv(root / "timeseries" / f"{rid}.csv", index=False, na_rep="NaN")  # fmt: skip
    return sweep


def test_plot_script_and_report_on_a_tiny_fake_results_folder(tmp_path):
    fake_results(tmp_path)
    subprocess.run([sys.executable, str(ROOT / "scripts/plot_training_curves.py"), "--results-dir", str(tmp_path)], check=True, cwd=ROOT, capture_output=True)
    assert (tmp_path / "figures/fig1_style.png").stat().st_size > 10_000
    assert (tmp_path / "figures/alpha_variants.png").stat().st_size > 10_000

    df, missing = load_timeseries(tmp_path)
    assert missing == ["VGG4_lr0.01_bs512_m0_seed1"]
    assert list(df["factor"].drop_duplicates()) == ["lr", "batch_size", "momentum"]  # the row order of the figure

    analyse.analyse(tmp_path)
    report = (tmp_path / "report.md").read_text()
    assert "Expected but missing: 1" in report
    assert "as in the paper" not in report or "CIs apart" in report
    assert "beta_incr_max" in report
    # 7 runs of the 8 in the config have results (one checkpoint each is unstable); the baseline counts once
    assert "negative 4, left the basin 0, diverged 3" in report
    assert "Stable 7" in report


def test_compare_extremes_needs_separated_intervals():
    weak, strong = (500.0, 10.0, 3), (300.0, 10.0, 3)
    assert analyse.compare_extremes(weak, strong, "lower").endswith("as in the paper")
    assert analyse.compare_extremes(strong, weak, "lower").endswith("opposite to the paper")
    assert "not separated" in analyse.compare_extremes((500.0, 300.0, 3), strong, "lower")
    assert "no CI" in analyse.compare_extremes((500.0, float("nan"), 1), strong, "lower")
    assert "not computable" in analyse.compare_extremes((float("nan"), float("nan"), 0), strong, "lower")


@pytest.mark.slow
@pytest.mark.skipif(not (ROOT / "data/cifar-10-batches-py").exists(), reason="CIFAR-10 not downloaded")
def test_smoke_run_end_to_end(tmp_path):
    command = [sys.executable, str(ROOT / "scripts/run_sweep.py"), "--smoke", "--out-dir", str(tmp_path), "--device", "cpu", "--data-root", str(ROOT / "data")]
    subprocess.run(command, check=True, cwd=ROOT, timeout=600, capture_output=True)
    csvs = sorted((tmp_path / "timeseries").glob("*.csv"))
    assert len(csvs) == 4  # 2 values per factor, sharing the baseline
    for path in csvs:
        df = pd.read_csv(path, dtype={"llc_flag": "str"})
        assert list(df.columns) == TIMESERIES_COLUMNS
        assert df["step"].max() == 300
        assert df["llc_flag"].notna().sum() == 3
        assert df["beta_incr_max"].notna().sum() == 2  # step 100 comes before the window of 181 is full
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "figures/fig1_style.png").exists()
    assert (tmp_path / "figures/alpha_variants.png").exists()

    # Resuming skips finished runs and leaves the results unchanged
    before = {path: path.read_text() for path in csvs}
    out = subprocess.run(command, check=True, cwd=ROOT, timeout=600, capture_output=True, text=True).stdout
    assert out.count("training finished, skip") == 4 and "LLC" not in out.replace("LLC_", "")
    assert all(path.read_text() == text for path, text in before.items())
    assert json.loads(next((tmp_path / "runs").iterdir()).joinpath("train.json").read_text())["window"] == 181
