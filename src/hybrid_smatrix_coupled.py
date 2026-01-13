#!/usr/bin/env python3
"""
Hybrid Light-Trapping Model with Proper RT/S-matrix Coupling.

This module implements correct coupling between ray-tracing and S-matrix:
- Fresnel coefficients computed via S-matrix at each interface
- No constant TIR_prob: R_front(θ,λ) computed from S-matrix
- Wavelength and angle-dependent front reflection

Physical mechanism:
1. Light enters Si from EVA through textured front (S-matrix)
2. Propagates through Si with absorption
3. Reflects at back (S-matrix for Si→Al)
4. Returns through Si with absorption
5. Reflects at front (S-matrix for Si→SiNx→EVA direction) - NOT constant TIR_prob!
6. Repeat until intensity below threshold

Author: D. Barchiesi / Claude Code
Date: January 2025
"""

import numpy as np
from pathlib import Path
import sys
from numba import njit, prange

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_optical_data_csv
from src.optical.smatrix_numba import SMatrixSolverNumba
from src.optical.volumetric_scattering import VolumetricScatteringModel


# =============================================================================
# Numba-optimized kernels with wavelength-dependent front reflection
# =============================================================================

@njit(cache=True)
def _rt_coupled_single(alpha_cm, d_si_cm, path_factor, T_encap, R_back,
                       R_front, n_pass_max):
    """
    Numba-optimized single-wavelength ray-tracing with proper coupling.

    R_front is wavelength-dependent (from S-matrix), NOT constant TIR_prob.
    """
    A_single = 1.0 - np.exp(-alpha_cm * d_si_cm * path_factor)

    intensity = T_encap
    absorbed = 0.0

    for _ in range(n_pass_max):
        # Forward pass
        absorbed += intensity * A_single
        intensity *= (1.0 - A_single)

        # Back reflection (S-matrix)
        intensity *= R_back

        # Return pass
        absorbed += intensity * A_single
        intensity *= (1.0 - A_single)

        # Front reflection - PROPERLY COUPLED via S-matrix
        # R_front is wavelength-dependent, computed from Si→SiNx→EVA interface
        intensity *= R_front

        if intensity < 1e-8:
            break

    return min(max(absorbed, 0.0), 1.0)


@njit(parallel=True, cache=True)
def _rt_coupled_vectorized(alpha_cm, d_si_cm, path_factor, T_encap, R_back,
                           R_front, n_passes):
    """
    Numba-optimized vectorized ray-tracing with proper coupling.

    R_front is an array (wavelength-dependent), NOT a scalar.
    """
    n_wl = len(alpha_cm)
    A_Si = np.zeros(n_wl)

    for i in prange(n_wl):
        A_Si[i] = _rt_coupled_single(
            alpha_cm[i], d_si_cm, path_factor,
            T_encap[i], R_back[i], R_front[i], n_passes[i]
        )

    return A_Si


@njit(cache=True)
def _compute_n_passes_coupled(alpha_cm, d_si_cm, R_back_avg, R_front_avg, n_max):
    """
    Numba-optimized calculation of number of passes needed.

    Uses average R_front instead of constant TIR_prob.
    """
    n_wl = len(alpha_cm)
    n_passes = np.zeros(n_wl, dtype=np.int32)

    for i in range(n_wl):
        A_single = 1.0 - np.exp(-alpha_cm[i] * d_si_cm)

        if A_single > 0.99:
            n_passes[i] = 1
        else:
            # Estimate passes needed for 99% absorption
            loss_factor = (1.0 - A_single) * R_back_avg * R_front_avg
            if loss_factor > 0 and loss_factor < 1:
                n_est = -4.605 / np.log(loss_factor + 1e-10)
                n_passes[i] = min(max(int(np.ceil(n_est)), 1), n_max)
            else:
                n_passes[i] = n_max

    return n_passes


# =============================================================================
# Main coupled model class
# =============================================================================

class HybridSMatrixCoupled:
    """
    Hybrid light-trapping model with proper RT/S-matrix coupling.

    Key difference from HybridLightTrappingFast:
    - No constant TIR_prob parameter
    - R_front computed via S-matrix for Si→SiNx→EVA interface
    - Proper wavelength and angle dependence
    - Accounts for pyramid geometry in internal angle calculation

    Physics of pyramid texture:
    - {111} KOH-etched pyramids have facet angle of 54.74° from horizontal
    - Light at normal incidence hits facet at 54.74° local angle
    - After refraction: θ_Si = arcsin(sin(54.74°)/n_Si) ≈ 13.5°
    - After back reflection, light hits front at large angle (~41°)
    - This is above critical angle for Si→EVA (~25°), causing TIR

    This should match pure S-matrix results for IAM while providing
    efficient ray-tracing for light-trapping calculation.
    """

    def __init__(self,
                 si_thickness_um: float,
                 ar_glass: bool,
                 ar_glass_material: str,
                 ar_glass_d_nm: float,
                 d_glass_mm: float,
                 d_eva_um: float,
                 d_sinx_nm: float,
                 sigma_glass_nm: float,
                 sigma_sinx_nm: float,
                 n_passes_max: int,
                 # S-matrix solver parameters (required, no defaults)
                 coherence_length_um: float,
                 n_phase_points,  # int or 'auto'
                 min_phase_points: int,
                 max_phase_points: int,
                 diffuse_model: str,
                 eva_material: str,
                 pyramid_facet_angle_deg: float,
                 # Optical material files (required, no defaults)
                 optical_materials: dict = None,
                 # Volumetric scattering parameters (enabled by default, calibrated on SR)
                 enable_volumetric_scattering: bool = True,
                 S_tedlar_base: float = 183.4,
                 tio2_concentration: float = 0.25,
                 S_eva_base: float = 0.0,
                 eva_bubble_density: float = 0.0):
        """
        Parameters
        ----------
        si_thickness_um : float
            Silicon thickness in micrometers
        ar_glass : bool
            Whether to include AR coating on glass
        ar_glass_material : str
            CSV filename for AR glass optical data
        ar_glass_d_nm : float
            AR glass thickness in nm
        d_glass_mm : float
            Glass thickness in mm
        d_eva_um : float
            EVA thickness in µm
        d_sinx_nm : float
            SiNx ARC thickness in nm
        sigma_glass_nm : float
            Glass surface roughness in nm
        sigma_sinx_nm : float
            SiNx surface roughness in nm
        n_passes_max : int
            Maximum number of passes for ray-tracing
        coherence_length_um : float
            Coherence length for S-matrix solver (µm)
        n_phase_points : int or 'auto'
            Number of phase points for averaging
        min_phase_points : int
            Minimum phase points for 'auto' mode
        max_phase_points : int
            Maximum phase points for 'auto' mode
        diffuse_model : str
            Diffuse scattering model ('lambert', 'beckmann', 'none')
        eva_material : str
            CSV filename for EVA optical data (deprecated, use optical_materials)
        pyramid_facet_angle_deg : float
            Pyramid facet angle from horizontal. For {111} KOH pyramids: 54.74°
        optical_materials : dict, optional
            Dictionary of optical material CSV filenames:
            - 'glass': glass optical data (e.g., 'glass_vogt_703ppm.csv')
            - 'eva': EVA optical data (e.g., 'eva_uvt_vogt2016.csv')
            - 'sinx': SiNx ARC optical data (e.g., 'SINx_Vogt_3.csv')
            - 'silicon': Silicon optical data (e.g., 'silicon_green2008.csv')
            - 'aluminum': Aluminum reflector optical data (e.g., 'aluminum_johnson_christy.csv')
            If None, uses eva_material for EVA and default files for others.
        enable_volumetric_scattering : bool, optional
            Enable volumetric scattering in EVA and Tedlar (default True).
            When enabled, improves SR accuracy in NIR by ~37% (RMSE 5.7% -> 3.6%).
            NOTE: This affects SR calibration, NOT IAM calibration.
            IAM is dominated by Fresnel losses (S-matrix), scattering affects NIR light-trapping.
        S_tedlar_base : float, optional
            Base scattering coefficient for Tedlar (cm^-1), default 183.4 (calibrated).
            Higher values increase diffuse back-reflection. Calibrate on SR data.
        tio2_concentration : float, optional
            TiO2 pigment concentration in Tedlar (volume fraction), default 0.25 (calibrated).
            White Tedlar typically has 5-25% TiO2. Calibrate on SR data.
        S_eva_base : float, optional
            Base scattering coefficient for EVA (cm^-1), default 0.0.
            Non-zero if EVA has bubbles or additives.
        eva_bubble_density : float, optional
            Bubble density in EVA (cm^-3), default 0.0.
        """
        self.si_thickness_um = si_thickness_um
        self.d_si_cm = si_thickness_um * 1e-4
        self.ar_glass = ar_glass
        self.ar_glass_material = ar_glass_material
        self.ar_glass_d_nm = ar_glass_d_nm
        self.d_glass_mm = d_glass_mm
        self.d_eva_um = d_eva_um
        self.d_sinx_nm = d_sinx_nm
        self.sigma_glass_nm = sigma_glass_nm
        self.sigma_sinx_nm = sigma_sinx_nm
        self.n_passes_max = n_passes_max
        self.eva_material = eva_material
        self.pyramid_facet_angle_deg = pyramid_facet_angle_deg

        # Optical materials - REQUIRED, no fallback
        if optical_materials is None:
            raise ValueError(
                "optical_materials is required and must be a dict with keys:\n"
                "  'glass', 'eva', 'sinx', 'silicon', 'aluminum'\n"
                "Each value should be a CSV filename in data/optical_properties/materials/"
            )
        required_keys = {'glass', 'eva', 'sinx', 'silicon', 'aluminum'}
        missing_keys = required_keys - set(optical_materials.keys())
        if missing_keys:
            raise ValueError(f"optical_materials missing required keys: {missing_keys}")
        self.optical_materials = optical_materials

        # S-matrix solver parameters (required, no defaults)
        self.coherence_length_um = coherence_length_um
        self.n_phase_points = n_phase_points
        self.min_phase_points = min_phase_points
        self.max_phase_points = max_phase_points
        self.diffuse_model = diffuse_model

        # Volumetric scattering parameters
        self.enable_volumetric_scattering = enable_volumetric_scattering
        self.S_tedlar_base = S_tedlar_base
        self.tio2_concentration = tio2_concentration
        self.S_eva_base = S_eva_base
        self.eva_bubble_density = eva_bubble_density

        # Create scattering model if enabled
        if self.enable_volumetric_scattering:
            self._scattering_model = VolumetricScatteringModel(
                d_eva_um=self.d_eva_um,
                d_tedlar_um=350.0,  # Standard Tedlar thickness
                S_eva_base=self.S_eva_base,
                S_tedlar_base=self.S_tedlar_base,
                tio2_concentration=self.tio2_concentration,
                eva_bubble_density=self.eva_bubble_density,
                R_back=0.85  # Al reflectance
            )
        else:
            self._scattering_model = None

        # Cache for optical data
        self._opt_cache = {}

    def set_scattering_params(self, S_tedlar_base: float = None,
                               tio2_concentration: float = None,
                               S_eva_base: float = None,
                               eva_bubble_density: float = None,
                               enable: bool = None):
        """
        Update volumetric scattering parameters dynamically.

        Useful for calibration without recreating the model.

        Parameters
        ----------
        S_tedlar_base : float, optional
            Base scattering coefficient for Tedlar (cm^-1)
        tio2_concentration : float, optional
            TiO2 concentration in Tedlar (volume fraction)
        S_eva_base : float, optional
            Base scattering coefficient for EVA (cm^-1)
        eva_bubble_density : float, optional
            Bubble density in EVA (cm^-3)
        enable : bool, optional
            Enable/disable volumetric scattering
        """
        if enable is not None:
            self.enable_volumetric_scattering = enable

        if S_tedlar_base is not None:
            self.S_tedlar_base = S_tedlar_base
        if tio2_concentration is not None:
            self.tio2_concentration = tio2_concentration
        if S_eva_base is not None:
            self.S_eva_base = S_eva_base
        if eva_bubble_density is not None:
            self.eva_bubble_density = eva_bubble_density

        # Recreate scattering model with new parameters
        if self.enable_volumetric_scattering:
            self._scattering_model = VolumetricScatteringModel(
                d_eva_um=self.d_eva_um,
                d_tedlar_um=350.0,
                S_eva_base=self.S_eva_base,
                S_tedlar_base=self.S_tedlar_base,
                tio2_concentration=self.tio2_concentration,
                eva_bubble_density=self.eva_bubble_density,
                R_back=0.85
            )
        else:
            self._scattering_model = None

    def _load_optical_data(self, wavelengths):
        """Load and cache optical data from configured material files."""
        key = hash(wavelengths.tobytes())
        if key not in self._opt_cache:
            opt_data = {
                'glass': load_optical_data_csv(self.optical_materials['glass'], wavelengths),
                'eva': load_optical_data_csv(self.optical_materials['eva'], wavelengths),
                'sinx': load_optical_data_csv(self.optical_materials['sinx'], wavelengths),
                'si': load_optical_data_csv(self.optical_materials['silicon'], wavelengths),
                'al': load_optical_data_csv(self.optical_materials['aluminum'], wavelengths),
            }
            if self.ar_glass and self.ar_glass_material:
                opt_data['ar_glass'] = load_optical_data_csv(
                    self.ar_glass_material, wavelengths
                )
            self._opt_cache[key] = opt_data
        return self._opt_cache[key]

    def compute_encapsulation_smatrix(self, wavelengths, theta_ext_deg):
        """
        Compute transmission through encapsulation using S-matrix.

        Structure: Air → [AR] → Glass → EVA → SiNx → Si

        Returns
        -------
        T_into_Si : array
            Transmission into silicon
        theta_si_deg : float
            Angle in silicon after refraction
        opt : dict
            Optical data dictionary
        """
        n_wl = len(wavelengths)
        opt = self._load_optical_data(wavelengths)

        if self.ar_glass:
            n_ar = opt['ar_glass'][0] + 1j * opt['ar_glass'][1]

            n_complex = np.zeros((n_wl, 6), dtype=np.complex128)
            n_complex[:, 0] = 1.0  # Air
            n_complex[:, 1] = n_ar
            n_complex[:, 2] = opt['glass'][0] + 1j * opt['glass'][1]
            n_complex[:, 3] = opt['eva'][0] + 1j * opt['eva'][1]
            n_complex[:, 4] = opt['sinx'][0] + 1j * opt['sinx'][1]
            n_complex[:, 5] = opt['si'][0] + 1j * opt['si'][1]

            thicknesses = np.array([
                np.inf,
                self.ar_glass_d_nm * 1e-9,
                self.d_glass_mm * 1e-3,
                self.d_eva_um * 1e-6,
                self.d_sinx_nm * 1e-9,
                np.inf
            ])

            roughnesses = np.array([
                0.0,
                2e-9,
                self.sigma_glass_nm * 1e-9,
                5e-9,
                self.sigma_sinx_nm * 1e-9,
                0.0
            ])
        else:
            n_complex = np.zeros((n_wl, 5), dtype=np.complex128)
            n_complex[:, 0] = 1.0
            n_complex[:, 1] = opt['glass'][0] + 1j * opt['glass'][1]
            n_complex[:, 2] = opt['eva'][0] + 1j * opt['eva'][1]
            n_complex[:, 3] = opt['sinx'][0] + 1j * opt['sinx'][1]
            n_complex[:, 4] = opt['si'][0] + 1j * opt['si'][1]

            thicknesses = np.array([
                np.inf,
                self.d_glass_mm * 1e-3,
                self.d_eva_um * 1e-6,
                self.d_sinx_nm * 1e-9,
                np.inf
            ])

            roughnesses = np.array([
                0.0,
                self.sigma_glass_nm * 1e-9,
                5e-9,
                self.sigma_sinx_nm * 1e-9,
                0.0
            ])

        # S-matrix calculation
        solver = SMatrixSolverNumba(
            wavelengths_nm=wavelengths,
            angle_inc_deg=theta_ext_deg,
            coherence_length_um=self.coherence_length_um,
            n_phase_points=self.n_phase_points,
            diffuse_model=self.diffuse_model,
            min_phase_points=self.min_phase_points,
            max_phase_points=self.max_phase_points
        )

        result = solver.solve((n_complex, thicknesses, roughnesses),
                             polarization='average')

        # Transmission into Si = 1 - R - A_encapsulation
        T_into_Si = result['T']
        T_into_Si = np.clip(T_into_Si, 0, 1)

        # Angle in Si (Snell's law)
        n_si_avg = np.mean(np.real(opt['si'][0]))
        sin_si = np.sin(np.radians(theta_ext_deg)) / n_si_avg
        theta_si_deg = np.degrees(np.arcsin(np.clip(sin_si, -1, 1)))

        return T_into_Si, theta_si_deg, opt

    def compute_back_reflection(self, wavelengths, theta_si_deg, opt):
        """
        Compute back reflection at Si→Al interface using S-matrix.
        """
        n_wl = len(wavelengths)

        n_complex = np.zeros((n_wl, 2), dtype=np.complex128)
        n_complex[:, 0] = opt['si'][0] + 1j * opt['si'][1]
        n_complex[:, 1] = opt['al'][0] + 1j * opt['al'][1]

        thicknesses = np.array([np.inf, np.inf])
        roughnesses = np.array([0.0, 0.0])

        solver = SMatrixSolverNumba(
            wavelengths_nm=wavelengths,
            angle_inc_deg=theta_si_deg,
            coherence_length_um=self.coherence_length_um,
            n_phase_points=1,  # No phase averaging for metal
            diffuse_model=self.diffuse_model,
            min_phase_points=self.min_phase_points,
            max_phase_points=self.max_phase_points
        )

        result = solver.solve((n_complex, thicknesses, roughnesses),
                             polarization='average')

        return result['R']

    def compute_internal_angle_for_front_reflection(self, theta_ext_deg, opt):
        """
        Compute the effective internal angle for front reflection.

        For textured surfaces with pyramids, the internal angle for the return
        pass is different from the propagation angle due to pyramid geometry.

        Physics:
        1. External light at θ_ext hits pyramid facet at local angle
        2. After refraction into Si, light propagates at θ_Si
        3. After back reflection, light returns to front
        4. The angle relative to facet normal determines TIR

        For {111} pyramids (54.74°):
        - Normal incidence (θ_ext=0): local angle on facet = 54.74°
        - After refraction: θ_Si ≈ arcsin(sin(54.74°)/n_Si) ≈ 13.5°
        - Return angle to facet: ~41° (above critical angle → TIR)
        """
        n_si = np.mean(np.real(opt['si'][0]))
        facet_rad = np.radians(self.pyramid_facet_angle_deg)

        # For normal incidence, light hits facet at the facet angle
        # For oblique incidence, we need to account for the projection
        theta_ext_rad = np.radians(theta_ext_deg)

        # Local angle on facet (simplified: assume one dominant facet)
        # This is the angle between incident ray and facet normal
        theta_local_rad = facet_rad - theta_ext_rad

        # Angle propagation in Si after first refraction
        sin_theta_Si = np.sin(abs(theta_local_rad)) / n_si
        sin_theta_Si = np.clip(sin_theta_Si, -1, 1)
        theta_Si_prop = abs(np.arcsin(sin_theta_Si))

        # After back reflection, light returns toward front
        # The angle relative to the facet normal for internal reflection
        # is approximately: facet_angle - theta_Si_propagation
        theta_internal_to_facet = facet_rad - theta_Si_prop

        # Convert to degrees for S-matrix calculation
        theta_internal_deg = np.degrees(abs(theta_internal_to_facet))

        return theta_internal_deg

    def compute_front_reflection(self, wavelengths, theta_ext_deg, opt):
        """
        Compute front reflection at Si→SiNx→EVA interface using S-matrix.

        THIS IS THE KEY DIFFERENCE FROM HybridLightTrappingFast:
        - Light going FROM Si TOWARD EVA (reverse direction)
        - Accounts for pyramid geometry in internal angle calculation
        - Includes TIR for angles above critical angle
        - Wavelength-dependent via n(λ), k(λ)

        Returns R_front array (wavelength-dependent) replacing constant TIR_prob.
        """
        n_wl = len(wavelengths)

        # Compute effective internal angle accounting for pyramid geometry
        theta_internal_deg = self.compute_internal_angle_for_front_reflection(
            theta_ext_deg, opt
        )

        # Structure: Si (semi-inf) → SiNx → EVA (semi-inf)
        # Light is coming FROM Si, going toward EVA
        n_complex = np.zeros((n_wl, 3), dtype=np.complex128)
        n_complex[:, 0] = opt['si'][0] + 1j * opt['si'][1]      # Si (incident medium)
        n_complex[:, 1] = opt['sinx'][0] + 1j * opt['sinx'][1]  # SiNx ARC
        n_complex[:, 2] = opt['eva'][0] + 1j * opt['eva'][1]    # EVA (exit medium)

        thicknesses = np.array([
            np.inf,                    # Si semi-infinite
            self.d_sinx_nm * 1e-9,     # SiNx thin film
            np.inf                     # EVA semi-infinite
        ])

        roughnesses = np.array([
            0.0,
            self.sigma_sinx_nm * 1e-9,
            5e-9
        ])

        # Check for TIR: critical angle for Si→EVA
        n_si = np.mean(np.real(opt['si'][0]))
        n_eva = np.mean(np.real(opt['eva'][0]))
        theta_crit = np.degrees(np.arcsin(n_eva / n_si)) if n_eva < n_si else 90.0

        if theta_internal_deg >= theta_crit:
            # Total Internal Reflection - all light reflected
            return np.ones(n_wl)

        # Below critical angle: compute R via S-matrix
        solver = SMatrixSolverNumba(
            wavelengths_nm=wavelengths,
            angle_inc_deg=theta_internal_deg,
            coherence_length_um=self.coherence_length_um,
            n_phase_points=self.n_phase_points,
            diffuse_model=self.diffuse_model,
            min_phase_points=self.min_phase_points,
            max_phase_points=self.max_phase_points
        )

        result = solver.solve((n_complex, thicknesses, roughnesses),
                             polarization='average')

        # R_front is the reflection coefficient from Si toward EVA
        R_front = result['R']

        # Ensure physical bounds
        return np.clip(R_front, 0, 1)

    def compute_absorption(self, wavelengths, theta_ext_deg=0):
        """
        Compute Si absorption using properly coupled RT/S-matrix.

        1. S-matrix for encapsulation transmission (T_encap)
        2. S-matrix for back reflection (R_back)
        3. S-matrix for front reflection (R_front) - NOT constant TIR_prob!
        4. Ray-tracing with these wavelength-dependent coefficients

        Parameters
        ----------
        wavelengths : array
            Wavelengths in nm
        theta_ext_deg : float
            External incidence angle

        Returns
        -------
        dict with A_Si, T_encap, R_back, R_front, n_passes arrays
        """
        wavelengths = np.atleast_1d(wavelengths).astype(np.float64)
        n_wl = len(wavelengths)

        # Step 1: S-matrix encapsulation
        T_encap, theta_si_deg, opt = self.compute_encapsulation_smatrix(
            wavelengths, theta_ext_deg
        )

        # Step 2: S-matrix back reflection
        R_back = self.compute_back_reflection(wavelengths, theta_si_deg, opt)

        # Step 3: S-matrix front reflection (replaces constant TIR_prob!)
        # Uses theta_ext_deg to compute internal angle accounting for pyramid geometry
        R_front = self.compute_front_reflection(wavelengths, theta_ext_deg, opt)

        # Step 4: Absorption coefficient
        k_si = opt['si'][1]
        alpha_cm = 4 * np.pi * k_si / (wavelengths * 1e-7)

        # Number of passes
        R_back_avg = np.mean(R_back)
        R_front_avg = np.mean(R_front)
        n_passes = _compute_n_passes_coupled(
            alpha_cm, self.d_si_cm, R_back_avg, R_front_avg, self.n_passes_max
        )

        # Path length correction for angle in Si
        cos_theta_si = np.cos(np.radians(theta_si_deg))
        path_factor = 1.0 / max(cos_theta_si, 0.1)

        # Coupled ray-tracing with wavelength-dependent R_front
        A_Si = _rt_coupled_vectorized(
            alpha_cm, self.d_si_cm, path_factor,
            np.ascontiguousarray(T_encap),
            np.ascontiguousarray(R_back),
            np.ascontiguousarray(R_front),  # NOT constant!
            n_passes
        )

        A_Si = np.clip(A_Si, 0, 1)

        # Apply volumetric scattering correction if enabled
        # This improves SR accuracy in NIR by accounting for Tedlar diffuse reflection
        A_Si_uncorrected = A_Si.copy()
        if self._scattering_model is not None:
            A_Si = self._scattering_model.apply_correction(A_Si, wavelengths)
            A_Si = np.clip(A_Si, 0, 1)

        return {
            'A_Si': A_Si,
            'A_Si_uncorrected': A_Si_uncorrected,
            'T_encap': T_encap,
            'R_back': R_back,
            'R_front': R_front,
            'n_passes': n_passes,
            'theta_si_deg': theta_si_deg,
            'volumetric_scattering_enabled': self.enable_volumetric_scattering
        }

    def compute_iam(self, angles_deg, wavelengths, n_rays=50):
        """
        Compute IAM using Monte Carlo pyramid facet redistribution (DEFAULT).

        This is the recommended method for SMaRT as it properly handles:
        1. Encapsulation losses via S-matrix (Fresnel + interference)
        2. Pyramid angular redistribution via Monte Carlo

        Parameters
        ----------
        angles_deg : list
            External incidence angles
        wavelengths : array
            Wavelengths in nm
        n_rays : int
            Number of Monte Carlo rays (default 50, sufficient for convergence)

        Returns
        -------
        iam : dict
            IAM values keyed by angle
        details : dict
            Detailed results including iam_encap and f_lt_ratio
        """
        return self.compute_iam_montecarlo(angles_deg, wavelengths, n_rays=n_rays)

    def compute_iam_simple(self, angles_deg, wavelengths):
        """
        Compute IAM using simple RT-based model (LEGACY).

        This method is kept for backward compatibility but is less accurate
        than compute_iam (Monte Carlo) or compute_iam_smatrix_based.
        """
        results = {}
        for theta in angles_deg:
            results[theta] = self.compute_absorption(wavelengths, theta)

        # Reference at normal incidence
        A_Si_0 = np.mean(results[0]['A_Si'])

        iam = {}
        for theta in angles_deg:
            A_Si_theta = np.mean(results[theta]['A_Si'])
            iam[theta] = A_Si_theta / A_Si_0 if A_Si_0 > 0 else 0

        return iam, results

    def compute_iam_smatrix_based(self, angles_deg, wavelengths):
        """
        Compute IAM using pure S-matrix for angular dependence.

        For IAM, the angular dependence comes primarily from the front interfaces,
        which is better captured by pure S-matrix than separated ray-tracing.

        This method computes A_Si directly from S-matrix for the full structure
        at each angle, matching the structure used in calibrate_iam_smatrix.py:
        Air → Glass → EVA → SiNx → Si → EVA → Tedlar → Air (8 layers)
        """
        opt = self._load_optical_data(wavelengths)
        n_wl = len(wavelengths)

        # Load Tedlar optical data
        tedlar_data = load_optical_data_csv('tedlar_n_k_220_2200_10nm.csv', wavelengths)
        n_tedlar = tedlar_data[0] + 1j * tedlar_data[1]

        # Build full structure matching calibrate_iam_smatrix.py:
        # Air → Glass → EVA → SiNx → Si → EVA → Tedlar → Air (8 layers)
        n_complex = np.zeros((n_wl, 8), dtype=np.complex128)
        n_complex[:, 0] = 1.0  # Air
        n_complex[:, 1] = opt['glass'][0] + 1j * opt['glass'][1]
        n_complex[:, 2] = opt['eva'][0] + 1j * opt['eva'][1]
        n_complex[:, 3] = opt['sinx'][0] + 1j * opt['sinx'][1]
        n_complex[:, 4] = opt['si'][0] + 1j * opt['si'][1]
        n_complex[:, 5] = opt['eva'][0] + 1j * opt['eva'][1]
        n_complex[:, 6] = n_tedlar
        n_complex[:, 7] = 1.0  # Air

        thicknesses = np.array([
            np.inf,                        # Air
            self.d_glass_mm * 1e-3,        # Glass
            self.d_eva_um * 1e-6,          # EVA front
            self.d_sinx_nm * 1e-9,         # SiNx
            self.si_thickness_um * 1e-6,   # Si
            self.d_eva_um * 1e-6,          # EVA back (same thickness as front)
            350e-6,                        # Tedlar (~350µm)
            np.inf                         # Air
        ])

        roughnesses = np.array([
            0.0,
            self.sigma_glass_nm * 1e-9,
            5e-9,
            self.sigma_sinx_nm * 1e-9,
            0.1e-9,
            1e-9,
            5e-9,
            0.0
        ])

        si_layer_idx = 4

        # Compute A_Si at each angle
        results = {}
        J_ref = None  # Weighted photocurrent reference

        # Load AM1.5G for weighting
        try:
            from pvlib.spectrum import get_reference_spectra
            am15g = get_reference_spectra(standard='ASTM G173-03')
            irr = np.interp(wavelengths, am15g.index.values, am15g['global'].values)
        except:
            from pvlib.spectrum import get_am15g
            am15g = get_am15g()
            irr = np.interp(wavelengths, am15g.index.values, am15g.values)

        iam = {}
        for aoi in angles_deg:
            solver = SMatrixSolverNumba(
                wavelengths_nm=wavelengths,
                angle_inc_deg=aoi,
                coherence_length_um=self.coherence_length_um,
                n_phase_points=self.n_phase_points,
                min_phase_points=self.min_phase_points,
                max_phase_points=self.max_phase_points,
                diffuse_model=self.diffuse_model
            )

            result = solver.solve((n_complex, thicknesses, roughnesses),
                                 polarization='average')

            A_Si = result['A_layers'][si_layer_idx, :]

            # Weighted photocurrent
            J_weighted = np.trapz(A_Si * irr * wavelengths, wavelengths)

            results[aoi] = {'A_Si': A_Si, 'J_weighted': J_weighted}

            if aoi == 0 or J_ref is None:
                J_ref = J_weighted

            iam[aoi] = J_weighted / J_ref if J_ref > 0 else 1.0

        return iam, results

    def compute_iam_smart(self, angles_deg, wavelengths):
        """
        Compute IAM using hybrid approach: Encapsulation (S-matrix) × LT enhancement (RT).

        Physics:
        - IAM_encap: Fresnel losses at air/glass/EVA/SiNx (dominates ~95%)
        - f_LT: Light-trapping ENHANCEMENT factor from pyramids (should be ~1.0)

        CORRECTED: f_LT = (A_Si/T_encap)(θ) / (A_Si/T_encap)(0)
        This isolates the pyramid contribution by dividing out encapsulation.

        For well-designed modules, f_LT ≈ 1.0 because pyramids redistribute
        light internally regardless of external angle.
        """
        # Step 1: Encapsulation IAM via S-matrix (planar, up to Si interface)
        iam_encap = self._compute_encapsulation_iam(angles_deg, wavelengths)

        # Step 2: Light-trapping ENHANCEMENT factor (not ratio of A_Si!)
        # f_LT = (A_Si / T_encap) normalized to θ=0
        # This isolates the Si absorption efficiency from encapsulation
        results_rt = {}
        for theta in angles_deg:
            results_rt[theta] = self.compute_absorption(wavelengths, theta)

        # At θ=0: Si absorption efficiency = A_Si / T_encap
        T_encap_0 = np.mean(results_rt[0]['T_encap'])
        A_Si_0 = np.mean(results_rt[0]['A_Si'])
        eta_Si_0 = A_Si_0 / T_encap_0 if T_encap_0 > 0 else 1.0

        # Light-trapping enhancement ratio
        f_lt_ratio = {}
        for theta in angles_deg:
            T_encap_theta = np.mean(results_rt[theta]['T_encap'])
            A_Si_theta = np.mean(results_rt[theta]['A_Si'])
            eta_Si_theta = A_Si_theta / T_encap_theta if T_encap_theta > 0 else 1.0
            f_lt_ratio[theta] = eta_Si_theta / eta_Si_0 if eta_Si_0 > 0 else 1.0

        # Step 3: Combined IAM = encapsulation × LT enhancement
        iam = {}
        for theta in angles_deg:
            iam[theta] = iam_encap[theta] * f_lt_ratio[theta]

        return iam, {'iam_encap': iam_encap, 'f_lt_ratio': f_lt_ratio, 'results_rt': results_rt}

    def _compute_encapsulation_iam(self, angles_deg, wavelengths):
        """
        Compute IAM for encapsulation using FULL 8-layer structure.

        Uses same structure AND AM1.5G weighting as compute_iam_smatrix_based
        for consistency:
        Air → Glass → EVA → SiNx → Si → EVA → Tedlar → Air

        Returns weighted photocurrent normalized to θ=0.
        """
        opt = self._load_optical_data(wavelengths)
        n_wl = len(wavelengths)

        # Load Tedlar optical data
        tedlar_data = load_optical_data_csv('tedlar_n_k_220_2200_10nm.csv', wavelengths)
        n_tedlar = tedlar_data[0] + 1j * tedlar_data[1]

        # Full 8-layer structure (same as compute_iam_smatrix_based)
        n_complex = np.zeros((n_wl, 8), dtype=np.complex128)
        n_complex[:, 0] = 1.0  # Air
        n_complex[:, 1] = opt['glass'][0] + 1j * opt['glass'][1]
        n_complex[:, 2] = opt['eva'][0] + 1j * opt['eva'][1]
        n_complex[:, 3] = opt['sinx'][0] + 1j * opt['sinx'][1]
        n_complex[:, 4] = opt['si'][0] + 1j * opt['si'][1]
        n_complex[:, 5] = opt['eva'][0] + 1j * opt['eva'][1]
        n_complex[:, 6] = n_tedlar
        n_complex[:, 7] = 1.0  # Air

        thicknesses = np.array([
            np.inf,                        # Air
            self.d_glass_mm * 1e-3,        # Glass
            self.d_eva_um * 1e-6,          # EVA front
            self.d_sinx_nm * 1e-9,         # SiNx
            self.si_thickness_um * 1e-6,   # Si
            self.d_eva_um * 1e-6,          # EVA back
            350e-6,                        # Tedlar
            np.inf                         # Air
        ])

        roughnesses = np.array([
            0.0,
            self.sigma_glass_nm * 1e-9,
            5e-9,
            self.sigma_sinx_nm * 1e-9,
            0.1e-9,
            1e-9,
            5e-9,
            0.0
        ])

        si_layer_idx = 4

        # Load AM1.5G for weighting (same as compute_iam_smatrix_based)
        try:
            from pvlib.spectrum import get_reference_spectra
            am15g = get_reference_spectra(standard='ASTM G173-03')
            irr = np.interp(wavelengths, am15g.index.values, am15g['global'].values)
        except Exception:
            from pvlib.spectrum import get_am15g
            am15g = get_am15g()
            irr = np.interp(wavelengths, am15g.index.values, am15g.values)

        J_ref = None
        iam_encap = {}

        for aoi in angles_deg:
            solver = SMatrixSolverNumba(
                wavelengths_nm=wavelengths,
                angle_inc_deg=float(aoi),
                coherence_length_um=self.coherence_length_um,
                n_phase_points=self.n_phase_points,
                min_phase_points=self.min_phase_points,
                max_phase_points=self.max_phase_points,
                diffuse_model=self.diffuse_model
            )

            result = solver.solve((n_complex, thicknesses, roughnesses),
                                 polarization='average')

            # Use weighted photocurrent (same as compute_iam_smatrix_based)
            A_Si = result['A_layers'][si_layer_idx, :]
            J_weighted = np.trapz(A_Si * irr * wavelengths, wavelengths)

            if aoi == 0 or J_ref is None:
                J_ref = J_weighted

            iam_encap[aoi] = J_weighted / J_ref if J_ref > 0 else 1.0

        return iam_encap

    def compute_iam_montecarlo(self, angles_deg, wavelengths, n_rays=100):
        """
        Compute IAM using Monte Carlo angular redistribution on pyramid facets.

        CORRECTED VERSION:
        - Uses encapsulation IAM from S-matrix (captures Fresnel correctly)
        - Adds MC pyramid redistribution as enhancement factor

        Physics:
        - Light at external angle θ_ext hits pyramid facets at different local angles
        - 4 facet orientations (45°, 135°, 225°, 315° azimuth) at 54.74° from horizontal
        - Each facet sees a different effective angle depending on θ_ext and azimuth
        - Monte Carlo averages over random pyramid orientations
        """
        # Step 1: Get encapsulation IAM (S-matrix based)
        iam_encap = self._compute_encapsulation_iam(angles_deg, wavelengths)

        # Step 2: MC pyramid redistribution for enhancement factor
        alpha_rad = np.radians(self.pyramid_facet_angle_deg)

        # Precompute facet normals (4 facets)
        facet_normals = []
        for i in range(4):
            phi = np.radians(45.0 + 90.0 * i)
            nx = np.sin(alpha_rad) * np.cos(phi)
            ny = np.sin(alpha_rad) * np.sin(phi)
            nz = np.cos(alpha_rad)
            facet_normals.append((nx, ny, nz))

        # Compute Si absorption efficiency (A_Si/T_encap) for each angle via MC
        eta_Si = {}
        results_detail = {}

        for theta_ext in angles_deg:
            theta_ext_rad = np.radians(theta_ext)

            # Incident direction (in x-z plane)
            inc_x = np.sin(theta_ext_rad)
            inc_z = -np.cos(theta_ext_rad)

            eta_samples = []

            # Monte Carlo over random pyramid orientations
            np.random.seed(42)
            for ray in range(n_rays):
                # Random azimuthal orientation of pyramid
                psi = np.random.uniform(0, 2 * np.pi)
                cos_psi = np.cos(psi)
                sin_psi = np.sin(psi)

                # Rotate incident direction
                dir_x = inc_x * cos_psi
                dir_y = inc_x * sin_psi
                dir_z = inc_z

                # Find which facet is hit (most negative dot product with normal)
                best_dot = 1.0
                best_facet = 0

                for i, (nx, ny, nz) in enumerate(facet_normals):
                    # Rotate facet normal by psi
                    nx_rot = nx * cos_psi - ny * sin_psi
                    ny_rot = nx * sin_psi + ny * cos_psi

                    dot = dir_x * nx_rot + dir_y * ny_rot + dir_z * nz
                    if dot < best_dot:
                        best_dot = dot
                        best_facet = i

                # Local angle on facet = angle between ray and facet normal
                cos_theta_local = -best_dot  # Negative because dot is negative for hit
                theta_local_deg = np.degrees(np.arccos(np.clip(cos_theta_local, 0, 1)))

                # Compute absorption at this local angle
                result = self.compute_absorption(wavelengths, theta_local_deg)
                A_Si = np.mean(result['A_Si'])
                T_encap = np.mean(result['T_encap'])

                # Si absorption efficiency = A_Si / T_encap
                eta = A_Si / T_encap if T_encap > 0 else 1.0
                eta_samples.append(eta)

            eta_Si[theta_ext] = np.mean(eta_samples)

            results_detail[theta_ext] = {
                'eta_Si_mean': eta_Si[theta_ext],
                'eta_Si_std': np.std(eta_samples),
                'n_rays': n_rays
            }

        # Light-trapping enhancement ratio (normalized to θ=0)
        eta_0 = eta_Si[0]
        f_lt_ratio = {theta: eta_Si[theta] / eta_0 if eta_0 > 0 else 1.0
                      for theta in angles_deg}

        # Step 3: Combined IAM = encapsulation × LT enhancement
        iam = {}
        for theta in angles_deg:
            iam[theta] = iam_encap[theta] * f_lt_ratio[theta]

        return iam, {'iam_encap': iam_encap, 'f_lt_ratio': f_lt_ratio,
                     'results_detail': results_detail}


# =============================================================================
# Factory function for easy instantiation from config
# =============================================================================

def create_coupled_model(config):
    """
    Create HybridSMatrixCoupled from YAML config.

    All parameters MUST be defined in the config file.
    Raises ValueError if any required parameter is missing.

    Parameters
    ----------
    config : dict
        Configuration dictionary from load_config()

    Returns
    -------
    HybridSMatrixCoupled
        Configured model instance

    Raises
    ------
    ValueError
        If any required parameter is missing from config
    """
    # Validate required config sections
    if 'light_trapping' not in config:
        raise ValueError("Missing 'light_trapping' section in config")
    if 'structure' not in config:
        raise ValueError("Missing 'structure' section in config")

    lt_config = config['light_trapping']
    if 'smatrix' not in lt_config:
        raise ValueError("Missing 'smatrix' section in config['light_trapping']")
    smatrix_cfg = lt_config['smatrix']

    # Helper function to get layer thickness (required)
    def get_layer_thickness(layers, name):
        for layer in layers:
            if layer['name'] == name:
                return layer['thickness_m']
        raise ValueError(f"Missing layer '{name}' in config['structure']['layers']")

    # Helper function to get required parameter
    def require(d, key, section_name):
        if key not in d:
            raise ValueError(f"Missing required parameter '{key}' in {section_name}")
        return d[key]

    layers = config['structure']['layers']

    # Get structure parameters (all required)
    d_si = get_layer_thickness(layers, 'silicon')
    d_glass = get_layer_thickness(layers, 'glass')
    d_eva = get_layer_thickness(layers, 'eva_front')
    d_sinx = get_layer_thickness(layers, 'arc_sinx')

    # Get AR glass parameters (required)
    ar_glass = require(lt_config, 'ar_glass', "config['light_trapping']")
    ar_glass_material = require(lt_config, 'ar_glass_material', "config['light_trapping']")
    ar_glass_d_nm = require(lt_config, 'ar_glass_d_nm', "config['light_trapping']")

    # Get optical materials (required)
    optical_materials = require(lt_config, 'optical_materials', "config['light_trapping']")
    # Validate required keys
    for key in ['glass', 'eva', 'sinx', 'silicon', 'aluminum']:
        if key not in optical_materials:
            raise ValueError(f"Missing '{key}' in config['light_trapping']['optical_materials']")

    # Get volumetric scattering parameters (optional, with defaults)
    vol_scat = lt_config.get('volumetric_scattering', {})
    enable_scattering = vol_scat.get('enabled', True)
    S_tedlar_base = vol_scat.get('S_tedlar_base', 100.0)
    tio2_concentration = vol_scat.get('tio2_concentration', 0.20)
    S_eva_base = vol_scat.get('S_eva_base', 0.0)
    eva_bubble_density = vol_scat.get('eva_bubble_density', 0.0)

    return HybridSMatrixCoupled(
        si_thickness_um=d_si * 1e6,
        ar_glass=ar_glass,
        ar_glass_material=ar_glass_material,
        ar_glass_d_nm=ar_glass_d_nm,
        d_glass_mm=d_glass * 1e3,
        d_eva_um=d_eva * 1e6,
        d_sinx_nm=d_sinx * 1e9,
        sigma_glass_nm=require(lt_config, 'sigma_glass_nm', "config['light_trapping']"),
        sigma_sinx_nm=require(lt_config, 'sigma_sinx_nm', "config['light_trapping']"),
        n_passes_max=require(lt_config, 'n_passes_max', "config['light_trapping']"),
        coherence_length_um=require(smatrix_cfg, 'coherence_length_um', "config['light_trapping']['smatrix']"),
        n_phase_points=require(smatrix_cfg, 'n_phase_points', "config['light_trapping']['smatrix']"),
        min_phase_points=require(smatrix_cfg, 'min_phase_points', "config['light_trapping']['smatrix']"),
        max_phase_points=require(smatrix_cfg, 'max_phase_points', "config['light_trapping']['smatrix']"),
        diffuse_model=require(smatrix_cfg, 'diffuse_model', "config['light_trapping']['smatrix']"),
        eva_material=require(lt_config, 'eva_material', "config['light_trapping']"),
        pyramid_facet_angle_deg=require(lt_config, 'pyramid_facet_angle_deg', "config['light_trapping']"),
        optical_materials=optical_materials,
        # Volumetric scattering parameters
        enable_volumetric_scattering=enable_scattering,
        S_tedlar_base=S_tedlar_base,
        tio2_concentration=tio2_concentration,
        S_eva_base=S_eva_base,
        eva_bubble_density=eva_bubble_density,
    )


if __name__ == '__main__':
    # Quick test
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    from config import load_config

    print("=" * 70)
    print("Testing HybridSMatrixCoupled with proper RT/S-matrix coupling")
    print("=" * 70)
    print()

    # Load config
    config = load_config('vogt_2022_iam')
    layers = config['structure']['layers']
    lt_config = config['light_trapping']
    smatrix_cfg = lt_config['smatrix']

    # Extract layer thicknesses
    d_si = next(l['thickness_m'] for l in layers if l['name'] == 'silicon') * 1e6
    d_glass = next(l['thickness_m'] for l in layers if l['name'] == 'glass') * 1e3
    d_eva = next(l['thickness_m'] for l in layers if l['name'] == 'eva_front') * 1e6
    d_sinx = next(l['thickness_m'] for l in layers if l['name'] == 'arc_sinx') * 1e9

    wavelengths = np.linspace(400, 1100, 20)

    model = HybridSMatrixCoupled(
        si_thickness_um=d_si,
        ar_glass=lt_config['ar_glass'],
        ar_glass_material=lt_config['ar_glass_material'],
        ar_glass_d_nm=lt_config['ar_glass_d_nm'],
        d_glass_mm=d_glass,
        d_eva_um=d_eva,
        d_sinx_nm=d_sinx,
        sigma_glass_nm=lt_config['sigma_glass_nm'],
        sigma_sinx_nm=lt_config['sigma_sinx_nm'],
        n_passes_max=lt_config['n_passes_max'],
        coherence_length_um=smatrix_cfg['coherence_length_um'],
        n_phase_points=smatrix_cfg['n_phase_points'],
        min_phase_points=smatrix_cfg['min_phase_points'],
        max_phase_points=smatrix_cfg['max_phase_points'],
        diffuse_model=smatrix_cfg['diffuse_model'],
        eva_material=lt_config['eva_material'],
        pyramid_facet_angle_deg=lt_config['pyramid_facet_angle_deg'],
        optical_materials=lt_config['optical_materials'],
    )

    # First, show the pyramid geometry calculation
    opt = model._load_optical_data(wavelengths)
    theta_internal_0 = model.compute_internal_angle_for_front_reflection(0.0, opt)
    theta_internal_60 = model.compute_internal_angle_for_front_reflection(60.0, opt)

    n_si = np.mean(np.real(opt['si'][0]))
    n_eva = np.mean(np.real(opt['eva'][0]))
    theta_crit = np.degrees(np.arcsin(n_eva / n_si))

    print("Pyramid geometry analysis:")
    print(f"  Facet angle: {model.pyramid_facet_angle_deg:.2f}°")
    print(f"  n_Si (mean): {n_si:.3f}")
    print(f"  n_EVA (mean): {n_eva:.3f}")
    print(f"  Critical angle (Si→EVA): {theta_crit:.2f}°")
    print()
    print(f"  θ_ext=0° → θ_internal for front reflection: {theta_internal_0:.2f}°")
    print(f"  θ_ext=60° → θ_internal for front reflection: {theta_internal_60:.2f}°")
    print()
    if theta_internal_0 > theta_crit:
        print("  → At θ_ext=0°: TIR (above critical angle)")
    else:
        print(f"  → At θ_ext=0°: Partial reflection (below critical)")
    print()

    result = model.compute_absorption(wavelengths, 0.0)

    print("Results at normal incidence:")
    print(f"  A_Si mean: {np.mean(result['A_Si']):.3f}")
    print(f"  T_encap mean: {np.mean(result['T_encap']):.3f}")
    print(f"  R_back mean: {np.mean(result['R_back']):.3f}")
    print(f"  R_front mean: {np.mean(result['R_front']):.3f}")
    print(f"  n_passes range: {result['n_passes'].min()}-{result['n_passes'].max()}")
    print()

    # Test IAM
    print("IAM test:")
    angles = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    iam, results = model.compute_iam(angles, wavelengths)

    for angle in angles:
        r = results[angle]
        print(f"  θ={angle:2d}°: IAM={iam[angle]:.4f}, R_front={np.mean(r['R_front']):.3f}")
