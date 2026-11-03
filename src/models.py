import torch
import torch.nn as nn

# VGG family vendored from Simsekli et al. 2020, src/models.py (MIT licence):
# https://github.com/umutsimsekli/Hausdorff-Dimension-and-Generalization
VGG_CFG = {
    "VGG4": [512, "M"], 
    "VGG6": [256, "M", 512, "M", 512, "M"],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

# filter_nums in original code
VGG_CLASSIFIER_INPUT = {"VGG4": 131072, "VGG6": 8192}


class VGG(nn.Module):
    """Exactly HD's VGG: conv + ReLU blocks and max pooling, one linear classifier, no batchnorm or dropout."""

    def __init__(self, vgg_name: str, with_bias: bool = True):
        super().__init__()
        self.features = self._make_layers(VGG_CFG[vgg_name], with_bias)
        self.classifier = nn.Linear(VGG_CLASSIFIER_INPUT[vgg_name], 10, bias=with_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        return self.classifier(out.view(out.size(0), -1))

    def _make_layers(self, cfg: list, with_bias: bool) -> nn.Sequential:
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=with_bias), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class DLN(nn.Module):
    """Deep linear network: f(x) = W_L ... W_2 W_1 x, with no nonlinearities or biases.

    widths: layer sizes [input_size, hidden1, ..., hiddenK, output_size], so there are K weight
    matrices with W_l of shape (d_l, d_{l-1}).
    """

    def __init__(self, widths: list[int], sigma: float = 1.0):
        super().__init__()
        if len(widths) < 2:
            raise ValueError("widths needs at least an input and an output size")

        self.widths = list(widths)
        self.sigma = sigma
        self.layers = self.make_layers()

    def make_layers(self) -> nn.ModuleList:
        layers = nn.ModuleList()
        for d_in, d_out in zip(self.widths[:-1], self.widths[1:]):
            layer = nn.Linear(d_in, d_out, bias=False)
            # Every entry ~ N(0, sigma^2), independent of width (as in LLC's dln.py)
            nn.init.normal_(layer.weight, std=self.sigma)
            layers.append(layer)
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def end_to_end(self) -> torch.Tensor:
        """Product matrix W_L ... W_1 of shape (d_L, d_0)."""
        W = self.layers[0].weight
        for layer in self.layers[1:]:
            W = layer.weight @ W
        return W
