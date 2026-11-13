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


def test_load_telemetry_csv_and_visual_twin_json(tmp_path):
    import json

    sim = BioreactorSimulator()
    csv_path = tmp_path / "batch.csv"
    csv_path.write_text(
        "time_h,biomass_g_L,substrate_g_L,dissolved_oxygen_g_L,product_g_L\n"
        "0.0,0.8,35.0,0.0078,0.0\n"
        "2.0,1.5,33.2,0.0062,0.2\n",
        encoding="utf-8",
    )
    gt_csv, obs_csv = sim.load_telemetry_file(csv_path)
    assert len(gt_csv["t"]) == 2
    assert np.isclose(gt_csv["CL"][1], 0.0062)
    assert np.allclose(obs_csv["X_obs"], gt_csv["X"])

    json_path = tmp_path / "visual_twin_manifest.json"
    json_payload = {
        "metadata": {"software": "bioprocess-visual-twin", "version": "2.6.0"},
        "telemetry": [
            {"time_h": 0.0, "X": 0.8, "S": 35.0, "DO_pct": 100.0, "P": 0.0},
            {"time_h": 4.0, "X": 2.4, "S": 31.0, "DO_pct": 50.0, "P": 0.45},
        ],
    }
    json_path.write_text(json.dumps(json_payload), encoding="utf-8")
    gt_json, obs_json = sim.load_telemetry_file(json_path)
    assert len(gt_json["t"]) == 2
    assert np.isclose(gt_json["t"][-1], 4.0)
    assert np.isclose(gt_json["CL"][0], sim.p.C_star)
    assert np.isclose(gt_json["CL"][1], 0.5 * sim.p.C_star)
    assert np.allclose(obs_json["CL_obs"], gt_json["CL"])
