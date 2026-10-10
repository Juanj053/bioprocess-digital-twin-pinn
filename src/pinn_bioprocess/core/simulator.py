"""
Bioreactor numerical integration and synthetic multirate sensor generator.

Generates:
1. Dense ground truth trajectory via stiff Radau / RK45 integration.
2. Sparse offline analytical observations (HPLC / Spectrophotometry) with Gaussian noise.
3. High-frequency online sensor observations (Optical DO probe) with electrochemical noise.
"""

from typing import Dict, Tuple
import numpy as np
import scipy.integrate as spi

from pinn_bioprocess.core.kinetics import BioprocessParameters, BioprocessKinetics


class BioreactorSimulator:
    """Simulates high-fidelity physical trajectories and generates multi-rate sensor datasets."""

    def __init__(self, params: BioprocessParameters | None = None):
        self.p = params or BioprocessParameters()
        self.kinetics = BioprocessKinetics(self.p)

    def simulate_dense(self, n_points: int = 401) -> Dict[str, np.ndarray]:
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
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
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
