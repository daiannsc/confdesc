"""confdesc: from conformer ensembles and Gaussian outputs to ML-ready descriptor tables."""
from .ensemble import aggregate, boltzmann_weights, select_conformers, split_ensemble
from .gaussian import GaussianResult, read_output, write_input
from .xyz import Molecule, read_multixyz, write_xyz

__version__ = "0.1.0"
