# SMaRT: S-Matrix and Ray-Tracing Optical Solver for PV Modules

[![License: Academic Free / Commercial Contact](https://img.shields.io/badge/License-Academic%20Free-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx/xxxxx-blue.svg)](https://doi.org/10.xxxx/xxxxx)

**SMaRT** (S-Matrix and Ray-Tracing) is a hybrid optical solver for encapsulated crystalline silicon photovoltaic modules. It combines wave-optics S-matrix formalism for planar thin-film structures with Monte Carlo ray-tracing for pyramidal surface textures.

## Key Features

- **Numerically stable**: Log-space field propagation enables depth-resolved generation profiles G(z) through 180 um silicon (where standard TMM overflows)
- **Automatic coherence detection**: Halton quasi-Monte Carlo phase averaging eliminates manual coherency specification
- **Interface roughness**: Debye-Waller scattering correction for realistic module interfaces
- **Pyramidal textures**: Monte Carlo ray-tracing coupled with S-matrix for textured surfaces
- **Non-periodic textures**: Supports pyramid size distributions (unlike periodic RCWA/S4)
- **Fast**: 6000x speedup vs RayFlare while maintaining 0.3% agreement on Jsc

## Performance Comparison

| Solver   | Method | Time per spectrum | Calibration (1000 evals) |
|----------|--------|-------------------|--------------------------|
| SMaRT    | S-matrix + RT | ~1 s | ~17 min |
| RayFlare | RCWA + RT | ~6000 s | ~70 days |
| S4/RCWA  | Rigorous | ~1-10 s | ~hours |

## Validation

Validated against experimental measurements:

1. **IEC 61853-2 IAM** (Vogt et al. 2022): RMSE = 0.21%
2. **Spectral Response**: RMSE = 5.0% (limited by IQE model, not SMaRT optics)
3. **Pyramidal textures** (Scheul et al. 2020): Agreement with RayFlare within 0.3%

## Installation

```bash
pip install numpy scipy numba pyyaml
git clone https://github.com/dbarchiesi/SMaRT.git
cd SMaRT
```

## Quick Start

```python
import numpy as np
from src.optical.smatrix_numba import SMatrixSolverNumba
from src.hybrid_smatrix_coupled import HybridSMatrixCoupled

# Load configuration
import yaml
with open('config/vogt_2022_iam.yaml') as f:
    config = yaml.safe_load(f)

# Planar module: S-matrix solver
wavelengths = np.linspace(300, 1200, 91)
solver = SMatrixSolverNumba(wavelengths, angle_inc_deg=30)
result = solver.solve(layers, polarization='average')

# Textured module: Hybrid S-matrix + ray-tracing
hybrid = HybridSMatrixCoupled(
    wavelengths=wavelengths,
    pyramid_angle_deg=54.74,  # <111> facet angle
    use_ray_tracing=True,
    n_rays=10000
)
result = hybrid.compute_optical_properties(angle_deg=30, structure_params=params)
```

## Project Structure

```
SMaRT/
├── src/
│   ├── optical/
│   │   ├── smatrix_numba.py       # Numba-optimized S-matrix solver
│   │   └── __init__.py
│   ├── hybrid_smatrix_coupled.py  # Hybrid S-matrix + ray-tracing
│   └── __init__.py
├── config/
│   ├── vogt_2022_iam.yaml         # IAM validation configuration
│   └── scheul_2020_pyramids.yaml  # Pyramidal texture configuration
├── data/
│   └── optical_properties/        # Material n,k data (CSV format)
├── examples/
│   ├── basic_planar.py            # Planar module example
│   └── textured_pyramid.py        # Textured module example
├── tests/
│   └── test_smatrix.py
├── docs/
│   └── theory.md                  # Mathematical formulation
├── LICENSE
├── requirements.txt
└── README.md
```

## Requirements

- Python >= 3.9
- NumPy >= 1.20
- SciPy >= 1.7
- Numba >= 0.55
- PyYAML >= 6.0

## Citation

If you use SMaRT in your research, please cite:

```bibtex
@article{Barchiesi2026_SMaRT,
  author = {Barchiesi, Dominique},
  title = {Hybrid S-Matrix and Ray-Tracing Optical Model for Encapsulated
           Crystalline Silicon Photovoltaic Modules: Implementation,
           Validation, and Comparison with Established Solvers},
  journal = {Solar Energy Materials and Solar Cells},
  year = {2026},
  doi = {10.xxxx/xxxxx},
  note = {Code available at https://github.com/dbarchiesi/SMaRT}
}
```

## License

**Academic/Research Use**: Free, with citation requirement.

**Commercial Use**: Contact the author for licensing.

See the [LICENSE](LICENSE) file for details.

## Author

**Dominique Barchiesi**
Light, Nanomaterials, Nanotechnologies (L2n), CNRS EMR 7004
Universite de Technologie de Troyes, France
dominique.barchiesi@utt.fr

Development assisted by Claude Code (Anthropic).

## Acknowledgments

- Vogt et al. (2022) for the IEC 61853-2 IAM dataset
- Scheul et al. (2020) for the pyramidal texture reflectance data
- Green (2008) for silicon optical constants
