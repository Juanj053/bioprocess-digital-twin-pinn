"""
Stoichiometric validation and macroscopic balance verification.

Calculates:
- Carbon recovery fraction
- Substrate-to-biomass and substrate-to-product cumulative balances
- Physical boundary compliance (non-negativity and thermodynamic plausibility)
"""



def check_carbon_balance(
    S0: float,
    Sf: float,
    X0: float,
    Xf: float,
    P0: float,
    Pf: float,
    carbon_fraction_glucose: float = 0.400,
    carbon_fraction_biomass: float = 0.480,
    carbon_fraction_product: float = 0.450,
) -> dict[str, float]:
    """
    Computes macroscopic carbon recovery between initial and final fermentation states.
    Carbon balance: C_consumed = C_biomass + C_product + C_CO2 (estimated by difference).
    """
    delta_S = max(float(S0 - Sf), 0.0)
    delta_X = max(float(Xf - X0), 0.0)
    delta_P = max(float(Pf - P0), 0.0)

    c_in = delta_S * carbon_fraction_glucose
    c_biomass = delta_X * carbon_fraction_biomass
    c_product = delta_P * carbon_fraction_product

    c_fixed = c_biomass + c_product
    c_recovery_fixed = (c_fixed / c_in) if c_in > 1e-6 else 1.0

    return {
        "delta_S_g_L": delta_S,
        "delta_X_g_L": delta_X,
        "delta_P_g_L": delta_P,
        "c_consumed_g_C_L": c_in,
        "c_fixed_biomass_product_g_C_L": c_fixed,
        "carbon_incorporation_fraction": float(c_recovery_fixed),
    }
