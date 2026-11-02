import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

# HD's train_vgg.py and eval_vgg.py. HD's utils.get_data has other constants, but the VGG pipeline does not use it.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalise(images):
    """uint8 images (N, H, W, C) to normalised float tensors (N, C, H, W), as ToTensor + Normalize. No augmentation."""
    x = torch.from_numpy(np.asarray(images)).permute(0, 3, 1, 2).float() / 255
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def load_cifar10(root="data", subset=None):
    """Returns CPU tensors (x_train, y_train, x_test, y_test); subset keeps the first images of each split."""
    train = CIFAR10(root, train=True, download=True)
    test = CIFAR10(root, train=False, download=True)
    return (
        normalise(train.data[:subset]),
        torch.tensor(train.targets[:subset]),
        normalise(test.data[:subset]),
        torch.tensor(test.targets[:subset]),
    )


@torch.no_grad()
def evaluate(model, x, y, device, batch_size=1000):
    """Accuracy and mean cross-entropy over the full set, in eval mode."""
    model.eval()
    correct, total_loss = 0, 0.0
    for start in range(0, len(x), batch_size):
        inputs, labels = x[start : start + batch_size].to(device), y[start : start + batch_size].to(device)
        outputs = model(inputs)
        correct += (outputs.argmax(1) == labels).sum().item()
        total_loss += F.cross_entropy(outputs, labels, reduction="sum").item()
    return correct / len(x), total_loss / len(x)
