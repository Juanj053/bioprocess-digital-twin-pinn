"""
Bioprocess Physics-Informed Neural Network Trainer.

Loss formulation:
L_total = L_data + lambda_ic * L_ic + lambda_phys * L_phys + lambda_balance * L_balance

Where:
- L_data: Multi-rate Mean Squared Error on sparse offline/online sensor observations.
- L_ic: Anchor loss enforcing the known initial conditions y(0) = [X0, S0, CL0, P0]^T.
- L_phys: Mean squared residual of the coupled ODE system evaluated at continuous collocation points:
    res_X = dX/dt - (mu - kd)*X
    res_S = dS/dt + (mu/Yxs + ms + qp/Yps)*X
    res_CL = dCL/dt - kla*(C* - CL) + qo2*X
    res_P = dP/dt - qp*X
- L_balance: Stoichiometric mass conservation regularizer.
"""

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from pinn_bioprocess.core.kinetics import BioprocessParameters
from pinn_bioprocess.models.neural_pinn import BioprocessPINN


class BioprocessPINNTrainer:
    """Trainer orchestrating physics loss, autograd collocation, and sensor fitting."""

    def __init__(
        self,
        model: BioprocessPINN,
        params: BioprocessParameters | None = None,
        n_collocation: int = 150,
        lambda_phys: float = 0.05,
        lambda_ic: float = 0.15,
        lambda_balance: float = 0.01,
        device: str | torch.device = "cpu",
    ):
        self.model = model.to(device)
        self.p = params or BioprocessParameters()
        self.device = torch.device(device)

        self.n_collocation = n_collocation
        self.lambda_phys = float(lambda_phys)
        self.lambda_ic = float(lambda_ic)
        self.lambda_balance = float(lambda_balance)

        # Collocation points continuously sampled across process horizon
        t_col = torch.linspace(0.0, self.p.t_final_h, n_collocation, dtype=torch.float32).view(-1, 1).to(self.device)
        self.t_col = nn.Parameter(t_col, requires_grad=True)

    def compute_losses(
        self,
        t_off: torch.Tensor,
        x_obs_norm: torch.Tensor,
        s_obs_norm: torch.Tensor,
        p_obs_norm: torch.Tensor,
        t_on: torch.Tensor,
        cl_obs_norm: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Calculates all loss components."""
        # 1. DATA LOSS (Normalized Space)
        pred_off = self.model(t_off)
        l_x = torch.mean((pred_off[:, 0:1] - x_obs_norm) ** 2)
        l_s = torch.mean((pred_off[:, 1:2] - s_obs_norm) ** 2)
        l_p = torch.mean((pred_off[:, 3:4] - p_obs_norm) ** 2)

        pred_on = self.model(t_on)
        l_cl = torch.mean((pred_on[:, 2:3] - cl_obs_norm) ** 2)

        l_data = l_x + l_s + l_p + l_cl

        # 2. INITIAL CONDITION LOSS
        t_zero = torch.tensor([[0.0]], dtype=torch.float32, device=self.device)
        X0_hat, S0_hat, CL0_hat, P0_hat = self.model.get_physical_states(t_zero)
        l_ic = (
            ((X0_hat - self.p.X0) / self.model.scale_X) ** 2 +
            ((S0_hat - self.p.S0) / self.model.scale_S) ** 2 +
            ((CL0_hat - self.p.CL0) / self.model.scale_CL) ** 2 +
            ((P0_hat - self.p.P0) / self.model.scale_P) ** 2
        ).squeeze()

        # 3. PHYSICS RESIDUAL LOSS (Autograd Differential Equations)
        X_col, S_col, CL_col, P_col = self.model.get_physical_states(self.t_col)

        # Automatic differentiation w.r.t physical time t [h]
        dX_dt = torch.autograd.grad(X_col, self.t_col, grad_outputs=torch.ones_like(X_col), create_graph=True)[0]
        dS_dt = torch.autograd.grad(S_col, self.t_col, grad_outputs=torch.ones_like(S_col), create_graph=True)[0]
        dCL_dt = torch.autograd.grad(CL_col, self.t_col, grad_outputs=torch.ones_like(CL_col), create_graph=True)[0]
        dP_dt = torch.autograd.grad(P_col, self.t_col, grad_outputs=torch.ones_like(P_col), create_graph=True)[0]

        # Kinetic rate expressions
        eps = 1e-6
        mu_col = self.p.mu_max * (S_col / (self.p.Ks + S_col + eps)) * (CL_col / (self.p.Ko + CL_col + eps))
        qp_col = self.p.alpha * mu_col + self.p.beta
        qo2_col = (mu_col / self.p.Yxo) + self.p.mo

        # ODE system equations
        rate_X = (mu_col - self.p.kd) * X_col
        rate_S = - ((mu_col / self.p.Yxs) + self.p.ms + (qp_col / self.p.Yps)) * X_col
        rate_CL = self.p.kla * (self.p.C_star - CL_col) - qo2_col * X_col
        rate_P = qp_col * X_col

        # Residuals in g/(L*h)
        res_X = dX_dt - rate_X
        res_S = dS_dt - rate_S
        res_CL = dCL_dt - rate_CL
        res_P = dP_dt - rate_P

        l_phys = torch.mean(res_X**2) + torch.mean(res_S**2) + torch.mean(res_CL**2) + torch.mean(res_P**2)

        # 4. CARBON BALANCE REGULARIZATION
        delta_S_cons = torch.relu(self.p.S0 - S_col)
        delta_X_prod = torch.relu(X_col - self.p.X0)
        carbon_excess = torch.relu((delta_X_prod / self.p.Yxs) - delta_S_cons)
        l_balance = torch.mean(carbon_excess**2)

        total_loss = l_data + self.lambda_ic * l_ic + self.lambda_phys * l_phys + self.lambda_balance * l_balance

        return {
            "total_loss": total_loss,
            "l_data": l_data,
            "l_ic": l_ic,
            "l_phys": l_phys,
            "l_balance": l_balance,
        }

    def fit(
        self,
        observations: dict[str, np.ndarray],
        epochs: int = 4000,
        lr: float = 2e-3,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Executes the PINN optimization training loop with Adam + Cosine Annealing.
        """
        # Convert observations to device tensors
        t_off_t = torch.tensor(observations["t_offline"], dtype=torch.float32, device=self.device).view(-1, 1)
        x_obs_t = torch.tensor(observations["X_obs"] / self.model.scale_X, dtype=torch.float32, device=self.device).view(-1, 1)
        s_obs_t = torch.tensor(observations["S_obs"] / self.model.scale_S, dtype=torch.float32, device=self.device).view(-1, 1)
        p_obs_t = torch.tensor(observations["P_obs"] / self.model.scale_P, dtype=torch.float32, device=self.device).view(-1, 1)

        t_on_t = torch.tensor(observations["t_online"], dtype=torch.float32, device=self.device).view(-1, 1)
        cl_obs_t = torch.tensor(observations["CL_obs"] / self.model.scale_CL, dtype=torch.float32, device=self.device).view(-1, 1)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)

        start_time = time.time()
        history = {
            "epoch": [],
            "total_loss": [],
            "l_data": [],
            "l_phys": [],
            "l_ic": [],
            "l_balance": [],
        }

        for ep in range(1, epochs + 1):
            optimizer.zero_grad()
            losses = self.compute_losses(t_off_t, x_obs_t, s_obs_t, p_obs_t, t_on_t, cl_obs_t)
            loss = losses["total_loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()

            if ep % 500 == 0 or ep == 1 or ep == epochs:
                history["epoch"].append(ep)
                history["total_loss"].append(float(loss.item()))
                history["l_data"].append(float(losses["l_data"].item()))
                history["l_phys"].append(float(losses["l_phys"].item()))
                history["l_ic"].append(float(losses["l_ic"].item()))
                history["l_balance"].append(float(losses["l_balance"].item()))

                if verbose and (ep % 1000 == 0 or ep == 1 or ep == epochs):
                    print(
                        f"Epoch {ep:5d}/{epochs} | "
                        f"Loss: {loss.item():.5f} | "
                        f"L_data: {losses['l_data'].item():.5f} | "
                        f"L_phys: {losses['l_phys'].item():.5f} | "
                        f"L_ic: {losses['l_ic'].item():.5f}"
                    )

        duration = time.time() - start_time
        return {
            "history": history,
            "training_time_seconds": duration,
            "final_loss": float(history["total_loss"][-1]),
        }

    def predict(self, t_eval: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluates model state predictions across arbitrary query timestamps."""
        self.model.eval()
        with torch.no_grad():
            t_t = torch.tensor(t_eval, dtype=torch.float32, device=self.device).view(-1, 1)
            X, S, CL, P = self.model.get_physical_states(t_t)
            return {
                "t": t_eval,
                "X": X.cpu().numpy().flatten(),
                "S": S.cpu().numpy().flatten(),
                "CL": CL.cpu().numpy().flatten(),
                "P": P.cpu().numpy().flatten(),
            }
