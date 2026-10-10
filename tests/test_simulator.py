"""Unit tests for the bioreactor simulator."""

import numpy as np
from pinn_bioprocess.core.simulator import BioreactorSimulator


def test_simulate_dense_trajectory():
    sim = BioreactorSimulator()
    gt = sim.simulate_dense(n_points=101)

    assert len(gt["t"]) == 101
    assert gt["t"][0] == 0.0
    assert gt["t"][-1] == 40.0

    # Biomass must increase overall
    assert gt["X"][-1] > gt["X"][0]

    # Substrate must be depleted
    assert gt["S"][-1] < gt["S"][0]

    # Product must increase
    assert gt["P"][-1] > gt["P"][0]


def test_generate_multirate_sensors():
    sim = BioreactorSimulator()
    gt, obs = sim.generate_multirate_sensors(n_offline=11, n_online=41, random_seed=42)

    assert len(obs["t_offline"]) == 11
    assert len(obs["X_obs"]) == 11
    assert len(obs["S_obs"]) == 11
    assert len(obs["P_obs"]) == 11

    assert len(obs["t_online"]) == 41
    assert len(obs["CL_obs"]) == 41

    # Concentrations must be non-negative
    assert np.all(obs["X_obs"] >= 0.0)
    assert np.all(obs["S_obs"] >= 0.0)
    assert np.all(obs["CL_obs"] >= 0.0)
    assert np.all(obs["P_obs"] >= 0.0)
