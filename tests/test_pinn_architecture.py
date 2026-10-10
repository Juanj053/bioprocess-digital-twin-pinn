"""Unit tests for PyTorch BioprocessPINN architecture."""

import torch
from pinn_bioprocess.models.neural_pinn import BioprocessPINN


def test_pinn_output_shape_and_non_negativity():
    model = BioprocessPINN(hidden_dim=32, num_layers=2, t_max=40.0)
    t = torch.linspace(0.0, 40.0, 50).view(-1, 1)

    norm_out = model(t)
    assert norm_out.shape == (50, 4)
    # Strict biological non-negativity enforced by softplus
    assert torch.all(norm_out >= 0.0)


def test_pinn_physical_states_scaling():
    model = BioprocessPINN(hidden_dim=32, num_layers=2, t_max=40.0)
    t = torch.tensor([[0.0], [20.0], [40.0]])

    X, S, CL, P = model.get_physical_states(t)
    assert X.shape == (3, 1)
    assert S.shape == (3, 1)
    assert CL.shape == (3, 1)
    assert P.shape == (3, 1)

    assert torch.all(X >= 0.0)
    assert torch.all(S >= 0.0)
    assert torch.all(CL >= 0.0)
    assert torch.all(P >= 0.0)


def test_pinn_higher_order_autograd():
    model = BioprocessPINN(hidden_dim=32, num_layers=2, t_max=40.0)
    t = torch.tensor([[10.0]], requires_grad=True)

    X, S, CL, P = model.get_physical_states(t)
    dX_dt = torch.autograd.grad(X, t, create_graph=True)[0]
    assert dX_dt is not None
    assert not torch.isnan(dX_dt)

    # Higher order derivative (curvature) should be non-zero / well-defined
    d2X_dt2 = torch.autograd.grad(dX_dt, t)[0]
    assert d2X_dt2 is not None
    assert not torch.isnan(d2X_dt2)
