"""Train an HD VGG on CIFAR-10 with plain SGD, recording weight projections during the final epoch.

Run with: uv run python src/cnn/train_vgg.py --out-dir results/runs/example
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, when run as a script

from cnn.data import default_device, evaluate, load_cifar10
from estimators import random_directions
from models import VGG, VGG_CFG

NUM_PROJECTIONS = 10


class ProjectionRecorder:
    """Stores the projections of each weight tensor (ndim >= 2, biases skipped as in HD) on fixed random directions.

    Weights are projected on the CPU in float64: increments are orders of magnitude smaller than
    the iterates, and a float32 dot product over a million weights would round them away.
    """

    def __init__(self, model, generator, num_projections=NUM_PROJECTIONS):
        self.weights = {name: p for name, p in model.named_parameters() if p.ndim >= 2}
        self.directions = {
            name: random_directions(p.numel(), num_projections, generator, dtype=torch.float64)
            for name, p in self.weights.items()
        }
        self.projections = {name: [] for name in self.weights}
        self.zero_fraction = []

    def _flat_weights(self):
        return {name: p.detach().flatten().cpu().clone() for name, p in self.weights.items()}

    def start(self):
        self.previous = self._flat_weights()

    def record(self):
        current = self._flat_weights()
        for name, w in current.items():
            self.projections[name].append((self.directions[name] @ w.double()).numpy())
        self.zero_fraction.append([(current[name] == self.previous[name]).double().mean().item() for name in current])
        self.previous = current

    def save(self, path):
        """Each layer is T by K (steps by projections), zero_fraction is T by L (steps by layers)."""
        arrays = {f"layer/{name}": np.stack(rows) for name, rows in self.projections.items()}
        np.savez(path, zero_fraction=np.array(self.zero_fraction), **arrays)


def l2_penalty(model):
    return sum(p.pow(2).sum() for p in model.parameters())


def train(arch="VGG4", lr=0.01, batch_size=128, momentum=0.0, l2=0.0, epochs=50, seed=0, out_dir="results/runs/example", device=None, data_root="data", subset=None):
    """Trains one model and writes state_dict.pt, projections.npz and train.json to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or default_device())
    x_train, y_train, x_test, y_test = load_cifar10(data_root, subset)

    torch.manual_seed(seed)
    model = VGG(arch, with_bias=True).to(device)
    # Own generator, so that measuring never changes the training run
    recorder = ProjectionRecorder(model, torch.Generator().manual_seed(seed + 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    x, y = x_train.to(device), y_train.to(device)

    n = len(x)
    diverged = False
    model.train()
    for epoch in range(epochs):
        final_epoch = epoch == epochs - 1
        if final_epoch:
            recorder.start()
        perm = torch.randperm(n)
        running_loss, steps = 0.0, 0
        for start in range(0, n - batch_size + 1, batch_size):  # drop_last
            idx = perm[start : start + batch_size].to(device)
            loss = F.cross_entropy(model(x[idx]), y[idx])
            if l2:
                loss = loss + l2 * l2_penalty(model)
            if not torch.isfinite(loss):
                diverged = True
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            steps += 1
            if final_epoch:
                recorder.record()
        if diverged:
            print(f"diverged in epoch {epoch}")
            break
        print(f"epoch {epoch}: training objective {running_loss / steps:.4f}", flush=True)

    train_acc, train_loss = evaluate(model, x_train, y_train, device)
    test_acc, test_loss = evaluate(model, x_test, y_test, device)
    metrics = {
        "arch": arch,
        "lr": lr,
        "batch_size": batch_size,
        "momentum": momentum,
        "l2": l2,
        "epochs": epochs,
        "seed": seed,
        "num_train": n,
        "diverged": diverged,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "gap_acc": train_acc - test_acc,
        "gap_loss": test_loss - train_loss,
    }
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out_dir / "state_dict.pt")
    if not diverged:
        recorder.save(out_dir / "projections.npz")
    # Last, so that an existing train.json means the run is complete
    (out_dir / "train.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="VGG4", choices=sorted(VGG_CFG))
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--l2", type=float, default=0.0, help="adds l2 * sum of squared parameters to the training loss")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("results/runs/example"))
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--subset", type=int, default=None, help="keep only the first images of each split")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(train(**vars(parse_args())), indent=2))
