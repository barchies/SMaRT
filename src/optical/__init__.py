"""
S-Matrix optical solver for crystalline silicon PV modules.

Main class:
- SMatrixSolverNumba: High-level Numba-accelerated S-matrix solver

Features:
- Full S-matrix formalism (Lifeng Li 1996)
- Debye-Waller roughness correction at interfaces
- Adaptive phase averaging for incoherent layers
- Numba acceleration for performance

Usage:
    from src.optical.smatrix_numba import SMatrixSolverNumba

Author: D. Barchiesi
Date: December 2025
"""

from .smatrix_numba import SMatrixSolverNumba

__all__ = ['SMatrixSolverNumba']
