"""
Bioprocess kinetics and coupled ODE formulation for aerobic stirred tank bioreactors.

Implements:
- Dual-substrate/electron-acceptor Monod kinetics
- Pirt substrate consumption model
- Luedeking-Piret product formation kinetics
- Two-film volumetric gas-liquid mass transfer (kLa) and oxygen uptake rate (OUR)
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np


@dataclass
class BioprocessParameters:
    """Rigorous parameter container for 20 L aerobic stirred-tank fermentation."""
    
    # Kinetic parameters
    mu_max: float = 0.35        # Maximum specific growth rate [h^-1]
    Ks: float = 2.0             # Monod constant for carbon substrate [g/L]
    Ko: float = 0.0008          # Affinity constant for dissolved O2 [g/L] (0.8 mg/L)
    kd: float = 0.012           # Endogenous decay / cell death rate [h^-1]
    
    # Stoichiometric yields & maintenance
    Yxs: float = 0.48           # Biomass yield on substrate [g_X / g_S]
    ms: float = 0.020           # Pirt maintenance coefficient [g_S / (g_X * h)]
    Yxo: float = 1.10           # Biomass yield on oxygen [g_X / g_O2]
    mo: float = 0.015           # Respiration maintenance coefficient [g_O2 / (g_X * h)]
    Yps: float = 0.85           # Product yield on substrate [g_P / g_S]
    
    # Gas-liquid mass transfer
    kla: float = 65.0           # Volumetric mass transfer coefficient [h^-1]
    C_star: float = 0.0078      # Saturation DO at 30°C, 1.05 bar [g/L] (7.8 mg/L)
    
    # Luedeking-Piret product kinetics
    alpha: float = 0.15         # Growth-associated coefficient [g_P / g_X]
    beta: float = 0.010         # Non-growth-associated coefficient [g_P / (g_X * h)]
    
    # Initial conditions
    X0: float = 0.80            # Initial biomass [g/L]
    S0: float = 35.00           # Initial glucose [g/L]
    CL0: float = 0.0078         # Initial dissolved oxygen [g/L]
    P0: float = 0.00            # Initial product [g/L]
    t_final_h: float = 40.0     # Cultivation duration [h]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BioprocessParameters":
        """Instantiate parameters from a nested or flat dictionary."""
        kin = data.get("kinetics", {})
        yields = data.get("yields_and_maintenance", {})
        mt = data.get("mass_transfer", {})
        prod = data.get("product_formation", {})
        ic = data.get("initial_conditions", {})
        ves = data.get("vessel", {})

        return cls(
            mu_max=kin.get("mu_max", data.get("mu_max", 0.35)),
            Ks=kin.get("ks_glucose", data.get("Ks", 2.0)),
            Ko=kin.get("ko_oxygen", data.get("Ko", 0.0008)),
            kd=kin.get("kd_decay", data.get("kd", 0.012)),
            Yxs=yields.get("y_xs", data.get("Yxs", 0.48)),
            ms=yields.get("m_s", data.get("ms", 0.020)),
            Yxo=yields.get("y_xo", data.get("Yxo", 1.10)),
            mo=yields.get("m_o", data.get("mo", 0.015)),
            Yps=yields.get("y_ps", data.get("Yps", 0.85)),
            kla=mt.get("kla", data.get("kla", 65.0)),
            C_star=mt.get("c_star_do", data.get("C_star", 0.0078)),
            alpha=prod.get("alpha", data.get("alpha", 0.15)),
            beta=prod.get("beta", data.get("beta", 0.010)),
            X0=ic.get("x0_biomass", data.get("X0", 0.80)),
            S0=ic.get("s0_substrate", data.get("S0", 35.0)),
            CL0=ic.get("cl0_dissolved_oxygen", data.get("CL0", 0.0078)),
            P0=ic.get("p0_product", data.get("P0", 0.00)),
            t_final_h=ves.get("process_horizon_hours", data.get("t_final_h", 40.0)),
        )


class BioprocessKinetics:
    """Evaluates kinetic rates and ODE derivatives for the 20 L bioreactor system."""

    def __init__(self, params: BioprocessParameters | None = None):
        self.p = params or BioprocessParameters()

    def specific_growth_rate(self, S: float, CL: float) -> float:
        """
        Dual Monod specific growth rate mu [h^-1]:
        mu(S, CL) = mu_max * (S / (Ks + S)) * (CL / (Ko + CL))
        """
        if S <= 1e-6 or CL <= 1e-8:
            return 0.0
        return float(self.p.mu_max * (S / (self.p.Ks + S)) * (CL / (self.p.Ko + CL)))

    def specific_product_rate(self, mu: float, S: float) -> float:
        """Luedeking-Piret product rate q_P = alpha * mu + beta [g_P / (g_X * h)]."""
        if S <= 1e-6:
            return 0.0
        return float(self.p.alpha * mu + self.p.beta)

    def specific_oxygen_rate(self, mu: float, S: float) -> float:
        """Specific respiration rate q_O2 = (mu / Yxo) + mo [g_O2 / (g_X * h)]."""
        if S <= 1e-6:
            return 0.0
        return float((mu / self.p.Yxo) + self.p.mo)

    def evaluate_derivatives(self, t: float, y: np.ndarray | List[float]) -> List[float]:
        """
        Evaluates the 4 coupled ODE derivatives [dX/dt, dS/dt, dCL/dt, dP/dt].
        """
        X = max(float(y[0]), 0.0)
        S = max(float(y[1]), 0.0)
        CL = max(float(y[2]), 0.0)
        P = max(float(y[3]), 0.0)

        mu = self.specific_growth_rate(S, CL)
        qp = self.specific_product_rate(mu, S)
        qo2 = self.specific_oxygen_rate(mu, S)

        # 1. Biomass balance: dX/dt = (mu - kd) * X
        dX_dt = (mu - self.p.kd) * X

        # 2. Substrate balance: dS/dt = - (mu/Yxs + ms + qp/Yps) * X
        if S > 1e-6:
            dS_dt = - ((mu / self.p.Yxs) + self.p.ms + (qp / self.p.Yps)) * X
        else:
            dS_dt = 0.0

        # 3. Dissolved oxygen balance: dCL/dt = kla * (C* - CL) - qo2 * X
        dCL_dt = self.p.kla * (self.p.C_star - CL) - qo2 * X

        # 4. Product synthesis: dP/dt = qp * X
        dP_dt = qp * X

        return [dX_dt, dS_dt, dCL_dt, dP_dt]
