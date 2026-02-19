"""
Base Network
------------
Small 2-layer MLP that processes task inputs and produces a hidden state
used to query the memory bank.
"""

import torch
import torch.nn as nn


class BaseNet(nn.Module):
    """
    2-layer MLP backbone.

    Processes a flattened or timestep input and produces a hidden representation
    that is passed to the memory bank for querying.

    For sequence tasks the caller passes each timestep individually or the
    full sequence flattened; this module is kept simple and task-agnostic.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 64,
                 n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else output_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., input_dim]
        Returns:
            h: [..., output_dim]
        """
        return self.net(x)
