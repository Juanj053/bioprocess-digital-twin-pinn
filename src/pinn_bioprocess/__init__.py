"""
Physics-Informed Neural Network (PINN) Bioprocess Digital Twin.

Hybrid gray-box modeling framework embedding microbial kinetics,
mass transfer, and stoichiometric balance ODEs for real-time state estimation
and bioreactor soft-sensing.
"""

__version__ = "0.1.0"
__author__ = "Juan José Álvarez Morales"
__license__ = "MIT"

from pinn_bioprocess.core.kinetics import BioprocessKinetics
from pinn_bioprocess.core.simulator import BioreactorSimulator
from pinn_bioprocess.models.neural_pinn import BioprocessPINN
from pinn_bioprocess.training.trainer import BioprocessPINNTrainer

__all__ = [
    "BioprocessKinetics",
    "BioreactorSimulator",
    "BioprocessPINN",
    "BioprocessPINNTrainer",
]
