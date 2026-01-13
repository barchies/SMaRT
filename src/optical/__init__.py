"""
S-Matrix optical solver for crystalline silicon PV modules.

Main classes:
- SMatrixSolverNumba: High-level Numba-accelerated S-matrix solver
- VolumetricScatteringModel: Kubelka-Munk model for Tedlar backsheet scattering

Features:
- Full S-matrix formalism (Lifeng Li 1996)
- Debye-Waller roughness correction at interfaces
- Adaptive phase averaging for incoherent layers
- Numba acceleration for performance
- Volumetric scattering in Tedlar backsheet (TiO2 pigments)

Usage:
    from src.optical.smatrix_numba import SMatrixSolverNumba
    from src.optical.volumetric_scattering import VolumetricScatteringModel

Author: D. Barchiesi
Date: December 2025 / January 2026
"""

from .smatrix_numba import SMatrixSolverNumba
from .volumetric_scattering import VolumetricScatteringModel

__all__ = ['SMatrixSolverNumba', 'VolumetricScatteringModel']
