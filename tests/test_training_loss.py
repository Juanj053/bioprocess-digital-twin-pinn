"""Unit tests for BioprocessPINNTrainer and physical loss components."""

import torch
import numpy as np

from pinn_bioprocess.core.kinetics import BioprocessParameters
from pinn_bioprocess.core.simulator import BioreactorSimulator
from pinn_bioprocess.models.neural_pinn import BioprocessPINN
from pinn_bioprocess.training.trainer import BioprocessPINNTrainer


def test_loss_computation():
    params = BioprocessParameters()
    model = BioprocessPINN(hidden_dim=32, num_layers=2, t_max=40.0)
    trainer = BioprocessPINNTrainer(model=model, params=params, n_collocation=20)

    # Synthetic observations
    t_off = torch.tensor([[0.0], [20.0], [40.0]], dtype=torch.float32)
    x_obs = torch.tensor([[0.1], [0.5], [0.8]], dtype=torch.float32)
    s_obs = torch.tensor([[0.9], [0.4], [0.05]], dtype=torch.float32)
    p_obs = torch.tensor([[0.0], [0.2], [0.5]], dtype=torch.float32)

    t_on = torch.tensor([[0.0], [10.0], [20.0], [30.0], [40.0]], dtype=torch.float32)
    cl_obs = torch.tensor([[0.8], [0.5], [0.3], [0.4], [0.7]], dtype=torch.float32)

    losses = trainer.compute_losses(t_off, x_obs, s_obs, p_obs, t_on, cl_obs)

    assert "total_loss" in losses
    assert "l_data" in losses
    assert "l_phys" in losses
    assert "l_ic" in losses
    assert "l_balance" in losses

    for k, v in losses.items():
        assert not torch.isnan(v)
        assert v.item() >= 0.0


def test_short_training_optimization():
    sim = BioreactorSimulator()
    gt, obs = sim.generate_multirate_sensors(n_offline=5, n_online=10, random_seed=42)

    model = BioprocessPINN(hidden_dim=32, num_layers=2, t_max=40.0)
    trainer = BioprocessPINNTrainer(model=model, n_collocation=20)

    res = trainer.fit(obs, epochs=15, lr=0.01, verbose=False)

    assert res["training_time_seconds"] > 0.0
    assert len(res["history"]["total_loss"]) > 0
    assert not np.isnan(res["final_loss"])

    # Predict at ground truth timestamps
    pred = trainer.predict(gt["t"][:10])
    assert len(pred["X"]) == 10
    assert np.all(pred["X"] >= 0.0)
