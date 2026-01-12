"""
ArticleOptique - SMaRT Optical Modeling for c-Si PV Modules

SMaRT = S-Matrix and Ray-Tracing

This package provides the hybrid optical solver for crystalline silicon
photovoltaic modules with:
- Full S-matrix formalism (Lifeng Li 1996)
- Monte Carlo ray-tracing for pyramidal textures
- Debye-Waller roughness correction at interfaces
- Adaptive phase averaging for incoherent layers
- Numba acceleration for performance

Main modules:
    src.hybrid_smatrix_coupled - HybridSMatrixCoupled (main model class)
    src.optical.smatrix_numba - SMatrixSolverNumba (S-matrix solver)

Usage:
    from config import load_config
    from src.hybrid_smatrix_coupled import HybridSMatrixCoupled

    config = load_config('vogt_2022_iam')
    model = HybridSMatrixCoupled(...)
    iam, details = model.compute_iam(angles, wavelengths)

Author: D. Barchiesi
Date: January 2026
"""

__version__ = "2.0.0"
