"""Utility functions for RT-DETR."""
import math
import torch
import torch.nn as nn

__all__ = ['inverse_sigmoid', 'bias_init_with_prob', 'MLP']


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Logit (inverse sigmoid) function."""
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def bias_init_with_prob(prior_prob: float) -> float:
    """Compute bias initialisation value such that sigmoid(bias) = prior_prob."""
    return math.log(prior_prob / (1 - prior_prob))


class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
