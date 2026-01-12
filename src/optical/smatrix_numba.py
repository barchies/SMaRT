"""
Formalisme S-Matrix (Scattering Matrix) de Lifeng Li - Version Numba

Version optimisée pour Numba JIT compilation.
Toutes les fonctions numériques sont compilables par Numba.

Références:
- L. Li, J. Opt. Soc. Am. A 13, 1024-1035 (1996)
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, Dict, Optional

POL_TE = 0
POL_TM = 1
POL_AVERAGE = 2
DIFFUSE_LAMBERT = 0
DIFFUSE_BECKMANN = 1
DIFFUSE_NONE = 2


@njit(cache=True)
def halton_sequence_nb(n_points: int, base: int) -> np.ndarray:
    """
    Generate Halton quasi-Monte Carlo sequence in given base.

    Halton (1960) low-discrepancy sequence for deterministic phase averaging.
    """
    result = np.zeros(n_points, dtype=np.float64)
    for i in range(n_points):
        f = 1.0
        r = 0.0
        index = i + 1  # Start from 1 to avoid 0
        while index > 0:
            f = f / base
            r = r + f * (index % base)
            index = index // base
        result[i] = r
    return result


@njit(cache=True)
def compute_auto_n_phase_points(thicknesses: np.ndarray, L_coh: float,
                                  min_points: int, max_points: int) -> int:
    """
    Calcule automatiquement le nombre optimal de points de phase pour le moyennage.

    Critères:
    1. Si toutes les couches sont cohérentes (d < L_coh): n_phase = 1
    2. Sinon, n_phase dépend du nombre de couches incohérentes:
       - 1 couche incohérente: 8-16 points (Halton converge rapidement)
       - 2+ couches: réduire car le nombre d'échantillons = n_points^min(n_inc, 2)
    3. Le ratio d/L_coh affecte la convergence:
       - d >> L_coh: phase très aléatoire, convergence rapide
       - d ~ L_coh: zone de transition, plus de points nécessaires

    Parameters
    ----------
    thicknesses : np.ndarray
        Épaisseurs des couches en mètres
    L_coh : float
        Longueur de cohérence en mètres
    min_points : int
        Nombre minimum de points
    max_points : int
        Nombre maximum de points

    Returns
    -------
    int
        Nombre optimal de points de phase
    """
    n_layers = len(thicknesses)
    n_incoherent = 0
    max_ratio = 0.0  # max(d/L_coh) pour couches incohérentes

    for i in range(1, n_layers - 1):  # Exclure première et dernière (semi-infinies)
        d = thicknesses[i]
        if np.isinf(d) or d <= 0:
            continue
        if d >= L_coh:
            n_incoherent += 1
            ratio = d / L_coh
            if ratio > max_ratio:
                max_ratio = ratio

    # Cas 1: Toutes les couches sont cohérentes
    if n_incoherent == 0:
        return 1

    # Cas 2: Calcul basé sur le régime de cohérence
    # Pour d >> L_coh (ratio >> 1): phase totalement aléatoire,
    # Halton converge en O(log(N)/N), 8 points suffisent
    # Pour d ~ L_coh (ratio ~ 1-10): zone de transition, plus de points

    if max_ratio > 100:
        # Très incohérent (ex: verre 3mm, L_coh=5μm -> ratio=600)
        base_points = 6
    elif max_ratio > 10:
        # Incohérent (ex: EVA 450μm -> ratio=90)
        base_points = 8
    else:
        # Zone de transition (ratio 1-10)
        base_points = 12

    # Ajuster selon le nombre de couches incohérentes
    # Pour n_inc couches, on échantillonne n_points^min(n_inc, 2) combinaisons
    if n_incoherent == 1:
        n_points = base_points
    elif n_incoherent == 2:
        # n_points^2 échantillons, réduire la base
        n_points = max(min_points, int(np.sqrt(base_points * 16)))  # ~11-13
    else:
        # 3+ couches: limiter à 8 points (8^2 = 64 échantillons)
        n_points = max(min_points, min(8, base_points))

    # Appliquer les bornes
    return max(min_points, min(n_points, max_points))


@njit(cache=True)
def halton_sequence_2d_nb(n_points: int) -> np.ndarray:
    """
    Generate 2D Halton sequence using bases 2 and 3.
    Returns array of shape (n_points, 2).
    """
    result = np.zeros((n_points, 2), dtype=np.float64)
    result[:, 0] = halton_sequence_nb(n_points, 2)
    result[:, 1] = halton_sequence_nb(n_points, 3)
    return result


@njit(cache=True)
def redheffer_star_product_nb(S_a: np.ndarray, S_b: np.ndarray) -> np.ndarray:
    """Produit étoile de Redheffer."""
    S_a11, S_a12, S_a21, S_a22 = S_a[0], S_a[1], S_a[2], S_a[3]
    S_b11, S_b12, S_b21, S_b22 = S_b[0], S_b[1], S_b[2], S_b[3]
    D = 1.0 - S_a22 * S_b11
    if np.abs(D) < 1e-15:
        D = 1e-15 + 0j
    D_inv = 1.0 / D
    S11 = S_a11 + S_a12 * S_b11 * S_a21 * D_inv
    S12 = S_a12 * S_b12 * D_inv
    S21 = S_b21 * S_a21 * D_inv
    S22 = S_b22 + S_b21 * S_a22 * S_b12 * D_inv
    return np.array([S11, S12, S21, S22], dtype=np.complex128)


@njit(cache=True)
def identity_smatrix_nb() -> np.ndarray:
    return np.array([0.0+0j, 1.0+0j, 1.0+0j, 0.0+0j], dtype=np.complex128)


@njit(cache=True)
def compute_angles_nb(n_list: np.ndarray, angle_inc: float) -> np.ndarray:
    n_layers = len(n_list)
    theta_list = np.empty(n_layers, dtype=np.complex128)
    n0 = n_list[0]
    sin_theta0 = np.sin(angle_inc)
    for i in range(n_layers):
        sin_theta = n0 * sin_theta0 / n_list[i]
        theta = np.arcsin(sin_theta)
        if np.real(np.cos(theta)) < 0:
            theta = np.pi - theta
        theta_list[i] = theta
    return theta_list


@njit(cache=True)
def debye_waller_factor_nb(sigma: float, n1: complex, n2: complex,
                           theta1: complex, wavelength_m: float) -> float:
    if sigma <= 0:
        return 1.0
    cos1 = np.cos(theta1)
    sin1 = np.sin(theta1)
    cos2 = np.sqrt(1 - (n1 * sin1 / n2)**2)
    q = 2 * np.pi * np.abs(n1 * cos1 - n2 * cos2) / wavelength_m
    DW = np.exp(-2 * (q * sigma)**2)
    return np.real(DW)


@njit(cache=True)
def interface_smatrix_nb(n1: complex, n2: complex, theta1: complex, theta2: complex,
                         pol: int, roughness: float, wavelength_m: float) -> np.ndarray:
    cos1 = np.cos(theta1)
    cos2 = np.cos(theta2)
    if pol == POL_TE:
        num_r = n1 * cos1 - n2 * cos2
        denom = n1 * cos1 + n2 * cos2
        r12 = num_r / denom
        t12 = 2 * n1 * cos1 / denom
        r21 = -r12
        t21 = 2 * n2 * cos2 / denom
    else:
        num_r = n2 * cos1 - n1 * cos2
        denom = n2 * cos1 + n1 * cos2
        r12 = num_r / denom
        t12 = 2 * n1 * cos1 / denom
        r21 = -r12
        t21 = 2 * n2 * cos2 / denom
    if roughness > 0:
        DW = debye_waller_factor_nb(roughness, n1, n2, theta1, wavelength_m)
        r12 = r12 * DW
        r21 = r21 * DW
    return np.array([r12, t21, t12, r21], dtype=np.complex128)


@njit(cache=True)
def propagation_smatrix_nb(n: complex, theta: complex, d: float, wavelength_m: float) -> np.ndarray:
    k_z = 2 * np.pi * n * np.cos(theta) / wavelength_m
    phi = k_z * d
    phi_real = np.real(phi)
    phi_imag = np.imag(phi)
    if phi_imag > 30:
        phi_imag = 30.0
    elif phi_imag < -30:
        phi_imag = -30.0
    phi_safe = phi_real + 1j * phi_imag
    exp_phi = np.exp(1j * phi_safe)
    return np.array([0.0+0j, exp_phi, exp_phi, 0.0+0j], dtype=np.complex128)


@njit(cache=True)
def build_total_smatrix_nb(n_complex: np.ndarray, thicknesses: np.ndarray,
                           roughnesses: np.ndarray, theta_list: np.ndarray,
                           wavelength_m: float, pol: int) -> np.ndarray:
    n_layers = len(n_complex)
    S_total = identity_smatrix_nb()
    for i in range(n_layers - 1):
        roughness = roughnesses[i + 1]
        S_interface = interface_smatrix_nb(n_complex[i], n_complex[i + 1],
                                           theta_list[i], theta_list[i + 1],
                                           pol, roughness, wavelength_m)
        S_total = redheffer_star_product_nb(S_total, S_interface)
        d = thicknesses[i + 1]
        if not np.isinf(d):
            S_prop = propagation_smatrix_nb(n_complex[i + 1], theta_list[i + 1], d, wavelength_m)
            S_total = redheffer_star_product_nb(S_total, S_prop)
    return S_total


@njit(cache=True)
def build_smatrix_with_phase_nb(n_complex: np.ndarray, thicknesses: np.ndarray,
                                 roughnesses: np.ndarray, theta_list: np.ndarray,
                                 wavelength_m: float, pol: int,
                                 phase_overrides: np.ndarray,
                                 override_mask: np.ndarray) -> np.ndarray:
    n_layers = len(n_complex)
    S_total = identity_smatrix_nb()
    for i in range(n_layers - 1):
        roughness = roughnesses[i + 1]
        S_interface = interface_smatrix_nb(n_complex[i], n_complex[i + 1],
                                           theta_list[i], theta_list[i + 1],
                                           pol, roughness, wavelength_m)
        S_total = redheffer_star_product_nb(S_total, S_interface)
        d = thicknesses[i + 1]
        if not np.isinf(d):
            if override_mask[i + 1]:
                phi_imposed = phase_overrides[i + 1]
                n = n_complex[i + 1]
                theta = theta_list[i + 1]
                k_z = 2 * np.pi * n * np.cos(theta) / wavelength_m
                attenuation = np.exp(-np.imag(k_z) * d)
                exp_phi = attenuation * np.exp(1j * phi_imposed)
                S_prop = np.array([0.0+0j, exp_phi, exp_phi, 0.0+0j], dtype=np.complex128)
            else:
                S_prop = propagation_smatrix_nb(n_complex[i + 1], theta_list[i + 1], d, wavelength_m)
            S_total = redheffer_star_product_nb(S_total, S_prop)
    return S_total


@njit(cache=True)
def extract_RT_nb(S: np.ndarray, n_in: complex, n_out: complex,
                  theta_in: complex, theta_out: complex) -> Tuple[float, float]:
    R = np.abs(S[0])**2
    factor = np.real(n_out * np.cos(theta_out)) / np.real(n_in * np.cos(theta_in))
    T = np.abs(S[2])**2 * factor
    R = min(max(np.real(R), 0.0), 1.0)
    T = min(max(np.real(T), 0.0), 1.0)
    if R + T > 1.0:
        total = R + T
        R /= total
        T /= total
    return R, T


@njit(cache=True)
def is_coherent_nb(thickness: float, L_coh: float) -> bool:
    if np.isinf(thickness):
        return False
    return thickness < L_coh


@njit(cache=True)
def compute_diffuse_scattering_nb(DW: float, n1_real: float, n2_real: float,
                                   R_spec: float, T_spec: float,
                                   diffuse_model: int) -> Tuple[float, float, float, float]:
    """
    Calcule la diffusion BRDF/BTDF depuis interfaces rugueuses.

    DW = facteur Debye-Waller moyen (0-1), 1 = parfaitement spéculaire
    Returns: R_total, T_total, R_diffuse, T_diffuse
    """
    if DW >= 0.999:
        return R_spec, T_spec, 0.0, 0.0

    f_diff = 1 - DW
    R_specular = R_spec * DW
    T_specular = T_spec * DW
    E_diffuse = (R_spec + T_spec) * f_diff

    if diffuse_model == DIFFUSE_LAMBERT:
        # Modèle Lambertien: distribution isotrope
        if n2_real > n1_real:
            f_T = 0.7  # Plus de transmission vers milieu dense
        else:
            # Réflexion totale interne pour angles > critique
            if n1_real > n2_real:
                cos_crit = np.sqrt(1 - (n2_real / n1_real)**2)
            else:
                cos_crit = 0.0
            f_T = 0.5 * (1 + cos_crit)
        R_diffuse = E_diffuse * (1 - f_T)
        T_diffuse = E_diffuse * f_T
    elif diffuse_model == DIFFUSE_BECKMANN:
        # Modèle Beckmann: distribution plus peaked
        R_diffuse = E_diffuse * 0.4
        T_diffuse = E_diffuse * 0.6
    else:
        R_diffuse = 0.0
        T_diffuse = 0.0

    return R_specular + R_diffuse, T_specular + T_diffuse, R_diffuse, T_diffuse


@njit(cache=True)
def solve_single_wavelength_nb(n_complex: np.ndarray, thicknesses: np.ndarray,
                                roughnesses: np.ndarray, angle_inc: float,
                                wavelength_m: float, pol: int,
                                L_coh: float, n_phase_points: int,
                                diffuse_model: int = DIFFUSE_NONE) -> Tuple[float, float, float, float]:
    n_layers = len(n_complex)
    theta_list = compute_angles_nb(n_complex, angle_inc)
    has_incoherent = False
    incoherent_mask = np.zeros(n_layers, dtype=np.bool_)
    for i in range(1, n_layers - 1):
        if not is_coherent_nb(thicknesses[i], L_coh):
            has_incoherent = True
            incoherent_mask[i] = True

    if not has_incoherent:
        S_total = build_total_smatrix_nb(n_complex, thicknesses, roughnesses, theta_list, wavelength_m, pol)
        R, T = extract_RT_nb(S_total, n_complex[0], n_complex[-1], theta_list[0], theta_list[-1])
    else:
        # Halton quasi-Monte Carlo sequences for phase averaging (Halton 1960)
        R_sum = 0.0
        T_sum = 0.0
        n_inc = 0
        inc_indices = np.empty(n_layers, dtype=np.int64)
        for i in range(n_layers):
            if incoherent_mask[i]:
                inc_indices[n_inc] = i
                n_inc += 1

        if n_inc == 1:
            # 1D Halton sequence (base 2)
            halton_phases = halton_sequence_nb(n_phase_points, 2) * 2 * np.pi
            idx = inc_indices[0]
            phase_overrides = np.zeros(n_layers, dtype=np.float64)
            for phi in halton_phases:
                phase_overrides[idx] = phi
                S_total = build_smatrix_with_phase_nb(n_complex, thicknesses, roughnesses, theta_list,
                                                       wavelength_m, pol, phase_overrides, incoherent_mask)
                R_sum += np.abs(S_total[0])**2
                T_sum += np.abs(S_total[2])**2
            R_avg = R_sum / n_phase_points
            T_avg = T_sum / n_phase_points
        else:
            # Multi-dimensional Halton sequence (bases 2, 3, 5, 7, ...)
            n_samples = n_phase_points ** min(n_inc, 2)
            phase_overrides = np.zeros(n_layers, dtype=np.float64)
            primes = np.array([2, 3, 5, 7, 11, 13, 17, 19], dtype=np.int64)
            halton_sequences = np.zeros((n_inc, n_samples), dtype=np.float64)
            for j in range(min(n_inc, len(primes))):
                halton_sequences[j, :] = halton_sequence_nb(n_samples, primes[j])
            for s in range(n_samples):
                for j in range(n_inc):
                    phase_overrides[inc_indices[j]] = halton_sequences[j, s] * 2 * np.pi
                S_total = build_smatrix_with_phase_nb(n_complex, thicknesses, roughnesses, theta_list,
                                                       wavelength_m, pol, phase_overrides, incoherent_mask)
                R_sum += np.abs(S_total[0])**2
                T_sum += np.abs(S_total[2])**2
            R_avg = R_sum / n_samples
            T_avg = T_sum / n_samples

        factor = np.real(n_complex[-1] * np.cos(theta_list[-1])) / np.real(n_complex[0] * np.cos(theta_list[0]))
        T_avg *= factor
        R = min(max(R_avg, 0.0), 1.0)
        T = min(max(T_avg, 0.0), 1.0)

    # Calculer facteur Debye-Waller moyen pour diffusion
    DW_sum = 0.0
    n_interfaces = 0
    for i in range(n_layers - 1):
        sigma = roughnesses[i + 1]
        if sigma > 0:
            n1 = n_complex[i]
            n2 = n_complex[i + 1]
            theta1 = theta_list[i]
            DW = debye_waller_factor_nb(sigma, n1, n2, theta1, wavelength_m)
            DW_sum += DW
            n_interfaces += 1
    DW_avg = DW_sum / n_interfaces if n_interfaces > 0 else 1.0

    # Appliquer diffusion BRDF/BTDF
    if diffuse_model != DIFFUSE_NONE and DW_avg < 0.999:
        n1_real = np.real(n_complex[0])
        n2_real = np.real(n_complex[-1])
        R, T, R_diff, T_diff = compute_diffuse_scattering_nb(DW_avg, n1_real, n2_real, R, T, diffuse_model)
    else:
        R_diff = 0.0
        T_diff = 0.0

    return R, T, R_diff, T_diff


@njit(cache=True)
def compute_field_amplitudes_nb(n_complex: np.ndarray, thicknesses: np.ndarray,
                                 roughnesses: np.ndarray, theta_list: np.ndarray,
                                 wavelength_m: float, pol: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute forward and backward field amplitudes at layer entries.

    Returns
    -------
    E_forward : np.ndarray
        Forward field amplitude at entry of each layer
    E_backward : np.ndarray
        Backward field amplitude at entry of each layer
    """
    n_layers = len(n_complex)
    E_forward = np.zeros(n_layers, dtype=np.complex128)
    E_backward = np.zeros(n_layers, dtype=np.complex128)
    E_forward[0] = 1.0 + 0j

    # Build cumulative S-matrix and extract field amplitudes
    S_cumul = identity_smatrix_nb()

    for i in range(n_layers - 1):
        roughness = roughnesses[i + 1]
        S_int = interface_smatrix_nb(n_complex[i], n_complex[i + 1],
                                     theta_list[i], theta_list[i + 1],
                                     pol, roughness, wavelength_m)
        S_cumul = redheffer_star_product_nb(S_cumul, S_int)

        # E_forward[i+1] = amplitude at ENTRY of layer i+1 (after interface)
        E_forward[i + 1] = S_cumul[2]  # S21

        # Apply propagation for total S-matrix
        d = thicknesses[i + 1]
        if not np.isinf(d):
            S_prop = propagation_smatrix_nb(n_complex[i + 1], theta_list[i + 1], d, wavelength_m)
            S_cumul = redheffer_star_product_nb(S_cumul, S_prop)

    # Total reflection coefficient
    r_total = S_cumul[0]  # S11
    E_backward[0] = r_total

    # Compute backward amplitudes for intermediate layers
    for i in range(1, n_layers - 1):
        # Build S-matrix from layer i to end
        S_from_i = identity_smatrix_nb()
        for j in range(i, n_layers - 1):
            roughness = roughnesses[j + 1]
            S_int = interface_smatrix_nb(n_complex[j], n_complex[j + 1],
                                         theta_list[j], theta_list[j + 1],
                                         pol, roughness, wavelength_m)
            d = thicknesses[j + 1]
            if not np.isinf(d):
                S_prop = propagation_smatrix_nb(n_complex[j + 1], theta_list[j + 1], d, wavelength_m)
                S_layer = redheffer_star_product_nb(S_int, S_prop)
            else:
                S_layer = S_int
            S_from_i = redheffer_star_product_nb(S_from_i, S_layer)
        E_backward[i] = E_forward[i] * S_from_i[0]  # S11

    return E_forward, E_backward


@njit(cache=True)
def compute_E_squared_at_z_nb(z_local: float, d: float,
                               E_plus_entry: complex, E_minus_exit: complex,
                               k_z: complex, log_threshold: float) -> float:
    """
    Compute |E(z)|² at a local position within a layer.

    Uses log-space for numerical stability in thick absorbing layers.

    Parameters
    ----------
    z_local : float
        Position from layer entry (m)
    d : float
        Layer thickness (m)
    E_plus_entry : complex
        Forward field amplitude at layer entry
    E_minus_exit : complex
        Backward field amplitude at layer exit
    k_z : complex
        z-component of wave vector (1/m)
    log_threshold : float
        Log threshold below which field is considered negligible

    Returns
    -------
    E_sq : float
        |E(z)|²
    """
    # E+(z) in log-space
    log_E_entry_sq = 2 * np.log(np.abs(E_plus_entry) + 1e-100)
    decay_plus = np.imag(k_z) * z_local
    log_E_plus_sq = log_E_entry_sq - 2 * decay_plus
    E_plus_negligible = (log_E_plus_sq < log_threshold)

    # E-(z) in log-space
    if d < np.inf and np.abs(E_minus_exit) > 1e-15:
        z_from_exit = d - z_local
        log_E_exit_sq = 2 * np.log(np.abs(E_minus_exit) + 1e-100)
        decay_minus = np.imag(k_z) * z_from_exit
        log_E_minus_sq = log_E_exit_sq - 2 * decay_minus
        E_minus_negligible = (log_E_minus_sq < log_threshold)
    else:
        E_minus_negligible = True
        log_E_minus_sq = log_threshold - 1

    # Case 1: Both negligible
    if E_plus_negligible and E_minus_negligible:
        return 0.0

    # Case 2: E- negligible (dominant case in absorbing layers)
    if E_minus_negligible:
        return np.exp(log_E_plus_sq) if not E_plus_negligible else 0.0

    # Case 3: E+ negligible (rare)
    if E_plus_negligible:
        return np.exp(log_E_minus_sq)

    # Case 4: Both significant → interference
    phase_plus = k_z * z_local
    phase_minus = k_z * (d - z_local)
    E_plus_z = E_plus_entry * np.exp(1j * phase_plus)
    E_minus_z = E_minus_exit * np.exp(1j * phase_minus)
    E_total = E_plus_z + E_minus_z
    return np.abs(E_total)**2


@njit(cache=True, parallel=True)
def solve_spectrum_nb(n_complex_all: np.ndarray, thicknesses: np.ndarray,
                      roughnesses: np.ndarray, wavelengths_m: np.ndarray,
                      angle_inc: float, pol: int,
                      L_coh: float, n_phase_points: int,
                      diffuse_model: int = DIFFUSE_NONE) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_wl = len(wavelengths_m)
    R = np.empty(n_wl, dtype=np.float64)
    T = np.empty(n_wl, dtype=np.float64)
    R_diff = np.empty(n_wl, dtype=np.float64)
    T_diff = np.empty(n_wl, dtype=np.float64)
    for i_wl in prange(n_wl):
        n_complex = n_complex_all[i_wl, :]
        wl_m = wavelengths_m[i_wl]
        R[i_wl], T[i_wl], R_diff[i_wl], T_diff[i_wl] = solve_single_wavelength_nb(
            n_complex, thicknesses, roughnesses, angle_inc, wl_m, pol,
            L_coh, n_phase_points, diffuse_model)
    return R, T, R_diff, T_diff


class SMatrixSolverNumba:
    """Solveur S-Matrix optimisé avec Numba."""

    def __init__(self, wavelengths_nm: np.ndarray, angle_inc_deg: float,
                 coherence_length_um: float, n_phase_points: int | str,
                 diffuse_model: str, min_phase_points: int, max_phase_points: int):
        """
        Parameters
        ----------
        wavelengths_nm : np.ndarray
            Longueurs d'onde en nm
        angle_inc_deg : float
            Angle d'incidence en degrés
        coherence_length_um : float
            Longueur de cohérence en micromètres
        n_phase_points : int or 'auto'
            Nombre de points pour le moyennage de phase.
            Si 'auto': calculé automatiquement selon l'épaisseur des couches.
        diffuse_model : str
            Modèle de diffusion: 'lambert', 'beckmann', ou 'none'
        min_phase_points : int
            Nombre minimum de points de phase (pour le mode 'auto')
        max_phase_points : int
            Nombre maximum de points de phase (pour le mode 'auto')
        """
        self.wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float64)
        self.wavelengths_m = self.wavelengths_nm * 1e-9
        self.angle_inc = np.radians(angle_inc_deg)
        self.n_wavelengths = len(self.wavelengths_nm)
        self.L_coh = coherence_length_um * 1e-6
        self._min_phase_points = min_phase_points
        self._max_phase_points = max_phase_points

        # Mode auto ou valeur fixe
        if n_phase_points == 'auto':
            self._n_phase_points_auto = True
            self._n_phase_points_fixed = max_phase_points  # Fallback si auto échoue
        else:
            self._n_phase_points_auto = False
            self._n_phase_points_fixed = int(n_phase_points)

        if diffuse_model == 'lambert':
            self.diffuse_model = DIFFUSE_LAMBERT
        elif diffuse_model == 'beckmann':
            self.diffuse_model = DIFFUSE_BECKMANN
        else:
            self.diffuse_model = DIFFUSE_NONE

    @property
    def n_phase_points(self):
        """Retourne le nombre de points de phase (fixe ou calculé)."""
        return self._n_phase_points_fixed

    def _get_n_phase_points(self, thicknesses: np.ndarray) -> int:
        """Calcule le nombre de points de phase optimal."""
        if self._n_phase_points_auto:
            return compute_auto_n_phase_points(thicknesses, self.L_coh,
                                               self._min_phase_points, self._max_phase_points)
        return self._n_phase_points_fixed

    def solve(self, layers, polarization: str = 'average') -> Dict[str, np.ndarray]:
        if isinstance(layers, tuple):
            n_complex_all, thicknesses, roughnesses = layers
        else:
            n_complex_all, thicknesses, roughnesses = self._layers_to_arrays(layers)

        # Calcul automatique du nombre de points de phase
        n_phase = self._get_n_phase_points(thicknesses)

        if polarization == 'TE':
            pol_code = POL_TE
        elif polarization == 'TM':
            pol_code = POL_TM
        else:
            pol_code = POL_AVERAGE

        if pol_code == POL_AVERAGE:
            R_TE, T_TE, Rd_TE, Td_TE = solve_spectrum_nb(
                n_complex_all, thicknesses, roughnesses,
                self.wavelengths_m, self.angle_inc, POL_TE,
                self.L_coh, n_phase, self.diffuse_model)
            R_TM, T_TM, Rd_TM, Td_TM = solve_spectrum_nb(
                n_complex_all, thicknesses, roughnesses,
                self.wavelengths_m, self.angle_inc, POL_TM,
                self.L_coh, n_phase, self.diffuse_model)
            R = (R_TE + R_TM) / 2
            T = (T_TE + T_TM) / 2
            R_diffuse = (Rd_TE + Rd_TM) / 2
            T_diffuse = (Td_TE + Td_TM) / 2
        else:
            R, T, R_diffuse, T_diffuse = solve_spectrum_nb(
                n_complex_all, thicknesses, roughnesses,
                self.wavelengths_m, self.angle_inc, pol_code,
                self.L_coh, n_phase, self.diffuse_model)

        A = 1 - R - T
        n_layers = len(thicknesses)

        # Calculer A_layers: distribuer A entre couches selon Beer-Lambert
        # (même logique que _solve_phase_averaged dans smatrix_solver.py)
        A_layers = np.zeros((n_layers, self.n_wavelengths))

        for i_wl in range(self.n_wavelengths):
            A_total = A[i_wl]
            if A_total < 1e-10:
                continue

            # Potentiel d'absorption de chaque couche
            A_potential = np.zeros(n_layers)
            for i in range(n_layers):
                if thicknesses[i] == 0 or np.isinf(thicknesses[i]):
                    # Semi-infinie: pas d'absorption dans le modèle couche
                    continue

                n_complex = n_complex_all[i_wl, i]
                k_ext = np.imag(n_complex)
                if k_ext > 1e-12:
                    d = thicknesses[i]
                    wl_m = self.wavelengths_m[i_wl]
                    alpha = 4 * np.pi * k_ext / wl_m
                    # Potentiel: 1 - exp(-alpha*d)
                    A_potential[i] = 1 - np.exp(-alpha * d)

            # Distribuer A_total proportionnellement
            sum_potential = np.sum(A_potential)
            if sum_potential > 1e-10:
                A_layers[:, i_wl] = A_potential * (A_total / sum_potential)
            else:
                # Si aucune couche n'absorbe, mettre tout dans la première non semi-infinie
                for i in range(n_layers):
                    if thicknesses[i] > 0 and not np.isinf(thicknesses[i]):
                        A_layers[i, i_wl] = A_total
                        break

        return {'R': R, 'T': T, 'A': A, 'A_layers': A_layers,
                'R_diffuse': R_diffuse, 'T_diffuse': T_diffuse}

    def _layers_to_arrays(self, layers) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_layers = len(layers)
        n_wl = self.n_wavelengths
        n_complex_all = np.empty((n_wl, n_layers), dtype=np.complex128)
        thicknesses = np.empty(n_layers, dtype=np.float64)
        roughnesses = np.empty(n_layers, dtype=np.float64)
        for i, layer in enumerate(layers):
            n_complex_all[:, i] = layer.n_complex
            thicknesses[i] = layer.thickness
            roughnesses[i] = getattr(layer, 'roughness', 0.0)
        return n_complex_all, thicknesses, roughnesses

    def compute_field_profile(self, layers, z_points: np.ndarray,
                               polarization: str = 'average') -> Dict[str, np.ndarray]:
        """
        Compute field profiles E(z,λ) and absorption rate Q(z,λ).

        Unified stable method for all layers (absorbing or not).
        Uses log-space to avoid underflow up to |E|² < 10⁻³⁰⁰.

        **Rigorous normalization**: Q(z) is normalized to guarantee that
        ∫Q(z)dz over each layer = A_layers[layer, λ] (exact absorption).
        Thus ∫∫Q(z)dz over the ENTIRE stack = A_total (perfect conservation).

        Parameters
        ----------
        layers : list
            List of SMatrixLayer objects
        z_points : np.ndarray
            Points z where to compute profiles (m)
        polarization : str
            'TE', 'TM', or 'average'

        Returns
        -------
        dict with keys:
            - 'z': np.ndarray, z points
            - 'E_squared': np.ndarray (n_z, n_wl), |E(z,λ)|²
            - 'Q_z': np.ndarray (n_z, n_wl), Q(z,λ) normalized (exact conservation)
            - 'A_layers': np.ndarray (n_layers, n_wl), absorption per layer
            - 'layer_indices': np.ndarray (n_z), layer index for each z
            - 'z_boundaries': np.ndarray, layer boundaries
        """
        z_points = np.asarray(z_points, dtype=np.float64)
        n_z = len(z_points)
        n_wl = self.n_wavelengths

        # Convert layers to arrays
        n_complex_all, thicknesses, roughnesses = self._layers_to_arrays(layers)
        n_layers = len(thicknesses)

        # Log threshold for negligible field
        LOG_E_SQ_THRESHOLD = -690.0  # exp(-690) ≈ 10⁻³⁰⁰

        # ================================================================
        # COMPUTE A_TOTAL AND A_LAYERS FOR EXACT NORMALIZATION
        # ================================================================
        result = self.solve(layers, polarization)
        A_layers_ref = result['A_layers']  # Reference (correct with phase averaging)

        # ================================================================
        # BUILD GEOMETRY
        # ================================================================
        layer_z_start = np.zeros(n_layers, dtype=np.float64)
        z_current = 0.0
        first_finite_idx = -1

        for i in range(n_layers):
            if np.isinf(thicknesses[i]):
                if first_finite_idx < 0:
                    layer_z_start[i] = -np.inf
                else:
                    layer_z_start[i] = z_current
            else:
                if first_finite_idx < 0:
                    first_finite_idx = i
                layer_z_start[i] = z_current
                z_current += thicknesses[i]

        # Build z_boundaries and layer assignment
        z_boundaries = [0.0]
        layer_at_boundary = []
        for i in range(n_layers):
            if layer_z_start[i] >= 0 and not np.isinf(layer_z_start[i]):
                if not np.isinf(thicknesses[i]):
                    z_boundaries.append(z_boundaries[-1] + thicknesses[i])
                    layer_at_boundary.append(i)
                elif layer_z_start[i] >= 0:
                    z_boundaries.append(np.inf)
                    layer_at_boundary.append(i)
                    break

        # Assign layer index to each z point
        layer_idx = np.zeros(n_z, dtype=np.int64)
        for i_z in range(n_z):
            z = z_points[i_z]
            for i_b in range(len(z_boundaries) - 1):
                z_min = z_boundaries[i_b]
                z_max = z_boundaries[i_b + 1]
                if i_b == len(z_boundaries) - 2:
                    # Last boundary: use <= to capture z_max
                    if z_min <= z <= z_max:
                        layer_idx[i_z] = layer_at_boundary[i_b]
                        break
                else:
                    if z_min <= z < z_max:
                        layer_idx[i_z] = layer_at_boundary[i_b]
                        break

        # ================================================================
        # COMPUTE E²(z) FOR EACH WAVELENGTH AND POLARIZATION
        # ================================================================
        if polarization == 'average':
            pols_to_compute = [POL_TE, POL_TM]
            E_sq_by_pol = {}
        else:
            pols_to_compute = [POL_TE if polarization == 'TE' else POL_TM]

        for pol in pols_to_compute:
            E_sq_temp = np.zeros((n_z, n_wl), dtype=np.float64)

            for i_wl in range(n_wl):
                wl_m = self.wavelengths_m[i_wl]
                n_complex = n_complex_all[i_wl, :]
                theta_list = compute_angles_nb(n_complex, self.angle_inc)

                # Compute field amplitudes
                E_fwd, E_bwd = compute_field_amplitudes_nb(
                    n_complex, thicknesses, roughnesses, theta_list, wl_m, pol
                )

                # Backward amplitude at exit of each layer
                E_bwd_exit = np.zeros(n_layers, dtype=np.complex128)
                for i_layer in range(n_layers - 1):
                    E_bwd_exit[i_layer] = E_bwd[i_layer + 1] if i_layer + 1 < n_layers else 0
                E_bwd_exit[-1] = 0

                # Compute E²(z) at each point
                for i_z in range(n_z):
                    i_layer = layer_idx[i_z]
                    if i_layer >= n_layers:
                        continue

                    z = z_points[i_z]
                    z_local = z - layer_z_start[i_layer]
                    d = thicknesses[i_layer]
                    n = n_complex[i_layer]
                    theta = theta_list[i_layer]

                    k_z = 2 * np.pi * n * np.cos(theta) / wl_m

                    E_sq = compute_E_squared_at_z_nb(
                        z_local, d,
                        E_fwd[i_layer], E_bwd_exit[i_layer],
                        k_z, LOG_E_SQ_THRESHOLD
                    )
                    E_sq_temp[i_z, i_wl] = E_sq

            if polarization == 'average':
                E_sq_by_pol[pol] = E_sq_temp.copy()
            else:
                E_sq_all = E_sq_temp.copy()

        # Average TE and TM if needed
        if polarization == 'average':
            E_sq_all = (E_sq_by_pol[POL_TE] + E_sq_by_pol[POL_TM]) / 2

        # ================================================================
        # RIGOROUS Q(z) NORMALIZATION
        # ================================================================
        Q_z_all = np.zeros((n_z, n_wl), dtype=np.float64)

        # Integrate E²(z) over each layer
        I_layer = np.zeros((n_layers, n_wl), dtype=np.float64)

        for i_layer in range(n_layers):
            mask = (layer_idx == i_layer)
            n_pts = np.sum(mask)
            if n_pts == 0:
                continue

            z_layer = z_points[mask]

            for i_wl in range(n_wl):
                E_sq_layer = E_sq_all[mask, i_wl]
                if n_pts >= 2:
                    I_layer[i_layer, i_wl] = np.trapz(E_sq_layer, z_layer)
                elif n_pts == 1:
                    if not np.isinf(thicknesses[i_layer]):
                        I_layer[i_layer, i_wl] = E_sq_layer[0] * thicknesses[i_layer]

        # Normalize Q(z) using A_layers_ref (correct reference)
        for i_layer in range(n_layers):
            mask = (layer_idx == i_layer)
            n_pts = np.sum(mask)
            if n_pts == 0 or np.isinf(thicknesses[i_layer]):
                continue

            for i_wl in range(n_wl):
                A_layer = A_layers_ref[i_layer, i_wl]
                if A_layer < 1e-10:
                    continue

                E_sq_layer = E_sq_all[mask, i_wl]
                I_raw = I_layer[i_layer, i_wl]

                if I_raw > 1e-100:
                    # Normalize so ∫Q dz = A_layer
                    Q_z_all[mask, i_wl] = A_layer * E_sq_layer / I_raw
                else:
                    # Fallback: uniform distribution
                    Q_z_all[mask, i_wl] = A_layer / thicknesses[i_layer]

        return {
            'z': z_points,
            'E_squared': E_sq_all,
            'Q_z': Q_z_all,
            'A_layers': A_layers_ref,
            'layer_indices': layer_idx,
            'z_boundaries': np.array(z_boundaries[:-1])
        }

    def compute_Q_z_simple(self, layers, A_layers: np.ndarray,
                            z_points: np.ndarray) -> np.ndarray:
        """
        FAST Q(z,λ) computation from A_layers (Beer-Lambert approximation).

        ~1000× faster than compute_field_profile() for thermal applications
        where average spatial profile suffices (no interference needed).

        Uses A_layers already computed by solve() (correct with phase averaging!)
        and distributes absorption according to Beer-Lambert: Q(z) ∝ exp(-α×z)

        Parameters
        ----------
        layers : list
            List of SMatrixLayer objects
        A_layers : np.ndarray
            Shape (n_layers, n_wl) from solve()
        z_points : np.ndarray
            Spatial points for Q(z)

        Returns
        -------
        Q_z : np.ndarray
            Shape (len(z_points), len(wavelengths))
            with ∫Q(z)dz = A_layers[i] for each layer
        """
        n_complex_all, thicknesses, _ = self._layers_to_arrays(layers)
        n_layers = len(thicknesses)
        n_z = len(z_points)
        n_wl = self.n_wavelengths

        Q_z = np.zeros((n_z, n_wl), dtype=np.float64)

        z_cumul = 0.0
        for i_layer in range(n_layers):
            d = thicknesses[i_layer]
            if np.isinf(d) or d <= 0:
                continue

            z_start = z_cumul
            z_end = z_cumul + d
            mask = (z_points >= z_start) & (z_points < z_end)

            if not np.any(mask):
                z_cumul = z_end
                continue

            z_layer = z_points[mask]
            z_local = z_layer - z_start

            for i_wl in range(n_wl):
                A_total_layer = A_layers[i_layer, i_wl]
                if A_total_layer < 1e-10:
                    continue

                wl_m = self.wavelengths_m[i_wl]
                n_complex = n_complex_all[i_wl, i_layer]
                k_ext = np.imag(n_complex)
                alpha = 4 * np.pi * k_ext / wl_m

                if alpha < 1e-10:
                    # Transparent: uniform absorption
                    Q_z[mask, i_wl] = A_total_layer / d
                else:
                    # Beer-Lambert: Q(z) = α × exp(-α×z) normalized
                    Q_profile = alpha * np.exp(-alpha * z_local)
                    A_beer_lambert = 1 - np.exp(-alpha * d)
                    if A_beer_lambert > 1e-100:
                        Q_z[mask, i_wl] = A_total_layer * Q_profile / A_beer_lambert
                    else:
                        Q_z[mask, i_wl] = A_total_layer / d

            z_cumul = z_end

        return Q_z
