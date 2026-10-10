"""Unit tests for bioprocess kinetics and ODE rates."""

import pytest
import numpy as np

from pinn_bioprocess.core.kinetics import BioprocessParameters, BioprocessKinetics


def test_default_parameters():
    params = BioprocessParameters()
    assert params.mu_max == 0.35
    assert params.Ks == 2.0
    assert params.kla == 65.0
    assert params.X0 == 0.80


def test_specific_growth_rate():
    kin = BioprocessKinetics()
    # At high substrate and DO saturation
    mu = kin.specific_growth_rate(S=35.0, CL=0.0078)
    assert 0.28 < mu < 0.35

    # Zero substrate should give 0.0
    assert kin.specific_growth_rate(S=0.0, CL=0.0078) == 0.0

    # Zero oxygen should give 0.0
    assert kin.specific_growth_rate(S=35.0, CL=0.0) == 0.0


def test_ode_derivatives_shape_and_direction():
    kin = BioprocessKinetics()
    y_init = [0.80, 35.0, 0.0078, 0.0]
    derivs = kin.evaluate_derivatives(0.0, y_init)

    assert len(derivs) == 4
    dX, dS, dCL, dP = derivs

    # In early exponential phase:
    assert dX > 0.0  # Biomass grows
    assert dS < 0.0  # Substrate consumed
    assert dP > 0.0  # Product is generated
