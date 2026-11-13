"""
Physics-Informed Neural Network (BioprocessPINN) architecture.

Key bioengineering features:
1. Infinitely differentiable C^inf smooth activations (Tanh / SiLU) enabling exact
   higher-order automatic differentiation via `torch.autograd.grad`.
2. Softplus output projection strictly enforcing biological non-negativity (X, S, CL, P >= 0).
3. Characteristic physical scaling mapping normalized latent states to physical concentrations (g/L).
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class BioprocessPINN(nn.Module):
    """
    Continuous neural surrogate for bioreactor state estimation:
    hat{y}(t) = [X(t), S(t), CL(t), P(t)]^T
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        t_max: float = 40.0,
        activation: str = "tanh",
        scale_X: float = 12.0,
        scale_S: float = 40.0,
        scale_CL: float = 0.010,
        scale_P: float = 5.0,
    ):
        super().__init__()
        self.t_max = float(t_max)

        # Physical dimensional scale constants [g/L]
        self.scale_X = float(scale_X)
        self.scale_S = float(scale_S)
        self.scale_CL = float(scale_CL)
        self.scale_P = float(scale_P)

        # Select activation function
        if activation.lower() == "silu":
            act_fn = nn.SiLU
        else:
            act_fn = nn.Tanh

        # Construct MLP backbone
        layers = []
        layers.append(nn.Linear(1, hidden_dim))
        layers.append(act_fn())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn())

        layers.append(nn.Linear(hidden_dim, 4))
        self.network = nn.Sequential(*layers)

        # Xavier/Glorot normal initialization tailored for smooth activations
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass in normalized space:
        Input: time tensor `t` [h] of shape (N, 1).
        Output: normalized states tensor of shape (N, 4) in [0, inf).
        """
        tau = t / self.t_max  # Dimensionless time tau in [0, 1]
        raw_out = self.network(tau)
        return F.softplus(raw_out)

    def get_physical_states(
        self, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transforms normalized network outputs into dimensional state concentrations in g/L.
        Returns: (X, S, CL, P)
        """
        norm_states = self.forward(t)
        X = norm_states[:, 0:1] * self.scale_X
        S = norm_states[:, 1:2] * self.scale_S
        CL = norm_states[:, 2:3] * self.scale_CL
        P = norm_states[:, 3:4] * self.scale_P
        return X, S, CL, P
