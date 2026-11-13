"""
Bioreactor numerical integration and synthetic multirate sensor generator.

Generates:
1. Dense ground truth trajectory via stiff Radau / RK45 integration.
2. Sparse offline analytical observations (HPLC / Spectrophotometry) with Gaussian noise.
3. High-frequency online sensor observations (Optical DO probe) with electrochemical noise.
"""

import csv
import json
from pathlib import Path

import numpy as np
import scipy.integrate as spi

from pinn_bioprocess.core.kinetics import BioprocessKinetics, BioprocessParameters


class BioreactorSimulator:
    """Simulates high-fidelity physical trajectories and generates multi-rate sensor datasets."""

    def __init__(self, params: BioprocessParameters | None = None):
        self.p = params or BioprocessParameters()
        self.kinetics = BioprocessKinetics(self.p)

    def simulate_dense(self, n_points: int = 401) -> dict[str, np.ndarray]:
        """
        Solves the coupled ODE system across [0, t_final] with high numerical accuracy.
        """
        t_span = (0.0, self.p.t_final_h)
        y0 = [self.p.X0, self.p.S0, self.p.CL0, self.p.P0]
        t_eval = np.linspace(0.0, self.p.t_final_h, n_points)

        sol = spi.solve_ivp(
            fun=self.kinetics.evaluate_derivatives,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method="Radau",
            rtol=1e-7,
            atol=1e-9,
        )

        if not sol.success:
            # Fallback to RK45 if stiff solver encounters singular Jacobian
            sol = spi.solve_ivp(
                fun=self.kinetics.evaluate_derivatives,
                t_span=t_span,
                y0=y0,
                t_eval=t_eval,
                method="RK45",
                rtol=1e-6,
                atol=1e-8,
            )

        return {
            "t": sol.t,
            "X": sol.y[0],
            "S": sol.y[1],
            "CL": sol.y[2],
            "P": sol.y[3],
        }

    def generate_multirate_sensors(
        self,
        n_offline: int = 11,
        noise_offline_std: float = 0.035,
        n_online: int = 41,
        noise_online_std: float = 0.020,
        random_seed: int = 42,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Generates ground truth alongside realistic multi-rate observations:
        - Offline: Biomass (OD600/DCW), Substrate (HPLC glucose), Product (ELISA/HPLC).
        - Online: Dissolved Oxygen (optical/polarographic probe).
        """
        rng = np.random.default_rng(random_seed)
        ground_truth = self.simulate_dense(n_points=401)

        t_span = (0.0, self.p.t_final_h)
        y0 = [self.p.X0, self.p.S0, self.p.CL0, self.p.P0]

        # 1. Sparse offline analytical sampling (e.g. every 4 hours)
        t_offline = np.linspace(0.0, self.p.t_final_h, n_offline)
        sol_off = spi.solve_ivp(
            fun=self.kinetics.evaluate_derivatives,
            t_span=t_span,
            y0=y0,
            t_eval=t_offline,
            method="Radau",
        )

        x_true_off = sol_off.y[0]
        s_true_off = sol_off.y[1]
        p_true_off = sol_off.y[3]

        X_obs = np.maximum(x_true_off * (1.0 + noise_offline_std * rng.standard_normal(len(t_offline))), 0.0)
        S_obs = np.maximum(s_true_off * (1.0 + noise_offline_std * rng.standard_normal(len(t_offline))), 0.0)
        P_obs = np.maximum(p_true_off * (1.0 + noise_offline_std * rng.standard_normal(len(t_offline))), 0.0)

        # 2. Continuous online DO probe sampling (e.g. every 1 hour)
        t_online = np.linspace(0.0, self.p.t_final_h, n_online)
        sol_on = spi.solve_ivp(
            fun=self.kinetics.evaluate_derivatives,
            t_span=t_span,
            y0=y0,
            t_eval=t_online,
            method="Radau",
        )

        cl_true_on = sol_on.y[2]
        CL_obs = np.maximum(cl_true_on * (1.0 + noise_online_std * rng.standard_normal(len(t_online))), 0.0)

        observations = {
            "t_offline": t_offline,
            "X_obs": X_obs,
            "S_obs": S_obs,
            "P_obs": P_obs,
            "t_online": t_online,
            "CL_obs": CL_obs,
        }

        return ground_truth, observations

    def load_telemetry_file(
        self, filepath: str | Path
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Load batch telemetry from CSV or a bioprocess-visual-twin JSON export."""
        path = Path(filepath)
        if path.suffix.lower() == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            records = data.get("telemetry") or data.get("history") or []
            if not records:
                raise ValueError(f"No telemetry or history records found in {path}")

            t_vals: list[float] = []
            x_vals: list[float] = []
            s_vals: list[float] = []
            cl_vals: list[float] = []
            p_vals: list[float] = []
            for rec in records:
                t_vals.append(float(rec.get("time_h", rec.get("t", 0.0))))
                x_vals.append(float(rec.get("X", rec.get("biomass_g_L", 0.0))))
                s_vals.append(float(rec.get("S", rec.get("substrate_g_L", 0.0))))
                p_vals.append(float(rec.get("P", rec.get("product_g_L", 0.0))))
                if "CL" in rec or "dissolved_oxygen_g_L" in rec:
                    cl_val = float(rec.get("CL", rec.get("dissolved_oxygen_g_L", self.p.C_star)))
                elif "DO_pct" in rec:
                    cl_val = (float(rec["DO_pct"]) / 100.0) * self.p.C_star
                else:
                    cl_val = self.p.C_star
                cl_vals.append(cl_val)
        else:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                raise ValueError(f"No telemetry rows found in {path}")
            t_vals = [float(r["time_h"]) for r in rows]
            x_vals = [float(r["biomass_g_L"]) for r in rows]
            s_vals = [float(r["substrate_g_L"]) for r in rows]
            cl_vals = [float(r["dissolved_oxygen_g_L"]) for r in rows]
            p_vals = [float(r["product_g_L"]) for r in rows]

        t_arr = np.asarray(t_vals, dtype=float)
        x_arr = np.asarray(x_vals, dtype=float)
        s_arr = np.asarray(s_vals, dtype=float)
        cl_arr = np.asarray(cl_vals, dtype=float)
        p_arr = np.asarray(p_vals, dtype=float)

        ground_truth = {
            "t": t_arr,
            "X": x_arr,
            "S": s_arr,
            "CL": cl_arr,
            "P": p_arr,
        }
        observations = {
            "t_offline": t_arr,
            "X_obs": x_arr,
            "S_obs": s_arr,
            "P_obs": p_arr,
            "t_online": t_arr,
            "CL_obs": cl_arr,
        }
        return ground_truth, observations
