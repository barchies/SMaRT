#!/usr/bin/env python3
"""
Modele de scattering volumique pour modules PV.

Le scattering volumique peut se produire dans:
- EVA: bulles d'air, additifs UV-stabilisants, cristallisation partielle
- Tedlar: pigments TiO2 pour la reflectance blanche
- Verre: inhomogeneites, inclusions

Modeles implementes:
1. Kubelka-Munk (deux flux): simple, efficace pour couches diffusantes
2. Four-flux: extension avec composantes collimees et diffuses
3. Henyey-Greenstein: fonction de phase pour scattering anisotrope

References:
- Kubelka & Munk (1931), Z. Tech. Phys.
- Vargas & Niklasson (1997), Applied Optics 36(22)
- McIntosh et al. (2009), Prog. Photovolt. 17, 191-197

Author: D. Barchiesi
Date: January 2026
"""

import numpy as np
from numba import njit
from typing import Tuple, Dict, Optional


# =============================================================================
# KUBELKA-MUNK MODEL (Two-flux)
# =============================================================================

@njit(cache=True)
def kubelka_munk_RT(K: float, S: float, d: float) -> Tuple[float, float]:
    """
    Calcul R et T par le modele Kubelka-Munk.

    Le modele KM resout les equations de transfert radiatif
    pour deux flux (avant et arriere) dans un milieu diffusant.

    Parameters
    ----------
    K : float
        Coefficient d'absorption (cm^-1)
    S : float
        Coefficient de scattering (cm^-1)
    d : float
        Epaisseur de la couche (cm)

    Returns
    -------
    R : float
        Reflectance diffuse
    T : float
        Transmittance diffuse
    """
    if S < 1e-10:
        # Pas de scattering: Beer-Lambert simple
        T = np.exp(-K * d)
        R = 0.0
        return R, T

    # Parametres KM
    a = (K + S) / S
    b = np.sqrt(a * a - 1.0)

    # Eviter overflow
    if b * S * d > 50:
        # Couche optiquement epaisse
        R = (a - b) / (a + b)
        T = 0.0
        return R, T

    sinh_bSd = np.sinh(b * S * d)
    cosh_bSd = np.cosh(b * S * d)

    denom = b * cosh_bSd + a * sinh_bSd

    if abs(denom) < 1e-15:
        return 0.0, 0.0

    R = sinh_bSd / denom
    T = b / denom

    return max(0.0, min(1.0, R)), max(0.0, min(1.0, T))


@njit(cache=True)
def kubelka_munk_from_nk(n: float, k: float, wavelength_nm: float,
                         d_cm: float, scatter_coeff_cm: float) -> Tuple[float, float]:
    """
    Calcul KM a partir des constantes optiques n, k.

    K = 4*pi*k/lambda (absorption)
    S = scatter_coeff (parametre ajustable)

    Parameters
    ----------
    n : float
        Indice de refraction (non utilise directement dans KM)
    k : float
        Coefficient d'extinction
    wavelength_nm : float
        Longueur d'onde (nm)
    d_cm : float
        Epaisseur (cm)
    scatter_coeff_cm : float
        Coefficient de scattering (cm^-1)

    Returns
    -------
    R, T : float
        Reflectance et transmittance diffuses
    """
    # Coefficient d'absorption
    K = 4 * np.pi * k / (wavelength_nm * 1e-7)  # cm^-1

    return kubelka_munk_RT(K, scatter_coeff_cm, d_cm)


# =============================================================================
# FOUR-FLUX MODEL
# =============================================================================

@njit(cache=True)
def four_flux_RT(K: float, S: float, d: float, n_layer: float, n_ambient: float,
                 theta_inc_rad: float) -> Tuple[float, float, float, float]:
    """
    Modele four-flux: separe composantes collimees et diffuses.

    Resout le systeme couple:
    - Flux collime avant (I+) et arriere (I-)
    - Flux diffus avant (J+) et arriere (J-)

    Parameters
    ----------
    K : float
        Coefficient d'absorption (cm^-1)
    S : float
        Coefficient de scattering (cm^-1)
    d : float
        Epaisseur (cm)
    n_layer : float
        Indice de refraction de la couche
    n_ambient : float
        Indice du milieu ambiant
    theta_inc_rad : float
        Angle d'incidence (rad)

    Returns
    -------
    Tc : float
        Transmittance collimee
    Td : float
        Transmittance diffuse
    Rc : float
        Reflectance collimee (Fresnel)
    Rd : float
        Reflectance diffuse
    """
    # Angle dans la couche (Snell)
    sin_t = n_ambient * np.sin(theta_inc_rad) / n_layer
    if abs(sin_t) > 1:
        # Reflexion totale
        return 0.0, 0.0, 1.0, 0.0

    cos_t = np.sqrt(1 - sin_t * sin_t)
    theta_t = np.arcsin(sin_t)

    # Coefficients de Fresnel a l'interface
    cos_i = np.cos(theta_inc_rad)
    r_s = (n_ambient * cos_i - n_layer * cos_t) / (n_ambient * cos_i + n_layer * cos_t)
    r_p = (n_layer * cos_i - n_ambient * cos_t) / (n_layer * cos_i + n_ambient * cos_t)
    R_fresnel = 0.5 * (r_s * r_s + r_p * r_p)

    # Transmittance collimee (Beer-Lambert avec scattering comme perte)
    path_length = d / cos_t
    Tc = (1 - R_fresnel) * np.exp(-(K + S) * path_length) * (1 - R_fresnel)

    # Partie diffusee convertie depuis collimee
    scattered_fraction = (1 - np.exp(-S * path_length))

    # Kubelka-Munk pour la partie diffuse
    R_km, T_km = kubelka_munk_RT(K, S, d)

    Td = (1 - R_fresnel) * scattered_fraction * T_km
    Rd = (1 - R_fresnel) * scattered_fraction * R_km + R_fresnel

    # Correction pour la reflexion interne
    Rc = R_fresnel

    return (max(0, min(1, Tc)), max(0, min(1, Td)),
            max(0, min(1, Rc)), max(0, min(1, Rd)))


# =============================================================================
# SCATTERING COEFFICIENTS FOR COMMON MATERIALS
# =============================================================================

def get_eva_scattering_coefficient(wavelength_nm: float,
                                    bubble_density: float = 0.0,
                                    additive_concentration: float = 0.0) -> float:
    """
    Estime le coefficient de scattering de l'EVA.

    Sources de scattering dans l'EVA:
    1. Bulles d'air microscopiques (defauts de lamination)
    2. Additifs UV-stabilisants (benzotriazoles, etc.)
    3. Cristallisation partielle du polymere

    Parameters
    ----------
    wavelength_nm : float
        Longueur d'onde (nm)
    bubble_density : float
        Densite de bulles (cm^-3), typique: 0 a 1e6
    additive_concentration : float
        Concentration d'additifs (fraction massique), typique: 0.001-0.01

    Returns
    -------
    S : float
        Coefficient de scattering (cm^-1)
    """
    S = 0.0

    # Scattering Rayleigh par additifs (proportionnel a 1/lambda^4)
    if additive_concentration > 0:
        # Rayleigh: S ~ 1/lambda^4
        S_rayleigh = additive_concentration * 1e4 * (500.0 / wavelength_nm) ** 4
        S += S_rayleigh

    # Scattering Mie par bulles (faiblement dependant de lambda)
    if bubble_density > 0:
        # Bulles typiques: 1-10 um de diametre
        # Section efficace ~ pi * r^2 pour grosses particules
        bubble_radius_cm = 5e-4  # 5 um
        sigma_bubble = np.pi * bubble_radius_cm ** 2
        S_mie = bubble_density * sigma_bubble
        S += S_mie

    return S


def get_tedlar_scattering_coefficient(wavelength_nm: float,
                                       tio2_concentration: float = 0.10) -> float:
    """
    Estime le coefficient de scattering du Tedlar blanc.

    Le Tedlar blanc contient des pigments TiO2 (rutile) qui diffusent
    fortement la lumiere, donnant l'aspect blanc opaque.

    Parameters
    ----------
    wavelength_nm : float
        Longueur d'onde (nm)
    tio2_concentration : float
        Concentration volumique de TiO2, typique: 0.05-0.15

    Returns
    -------
    S : float
        Coefficient de scattering (cm^-1)
    """
    # TiO2 rutile: particules ~200-300 nm, scattering Mie
    # Maximum de scattering vers 400-500 nm

    # Modele empirique base sur Vargas & Niklasson 1997
    # S(lambda) a un maximum vers 450 nm pour TiO2 rutile

    lambda_max = 450.0  # nm
    width = 200.0  # nm

    # Profil gaussien pour le scattering
    S_peak = tio2_concentration * 500.0  # cm^-1 au maximum

    S = S_peak * np.exp(-0.5 * ((wavelength_nm - lambda_max) / width) ** 2)

    # Minimum dans le NIR (scattering Rayleigh residuel)
    S_min = tio2_concentration * 50.0 * (500.0 / max(wavelength_nm, 500)) ** 2
    S = max(S, S_min)

    return S


# =============================================================================
# INTEGRATION WITH S-MATRIX MODEL
# =============================================================================

@njit(cache=True)
def apply_volumetric_scattering_nb(A_Si_coherent: np.ndarray,
                                    wavelengths_nm: np.ndarray,
                                    S_eva_cm: np.ndarray,
                                    S_tedlar_cm: np.ndarray,
                                    d_eva_cm: float,
                                    d_tedlar_cm: float,
                                    R_back: float) -> np.ndarray:
    """
    Applique une correction de scattering volumique a A_Si.

    Le scattering dans l'EVA et le Tedlar peut:
    1. Augmenter le chemin optique (light trapping)
    2. Redistribuer la lumiere vers des angles plus grands
    3. Augmenter la reflexion arriere effective

    Parameters
    ----------
    A_Si_coherent : array
        Absorption Si calculee par S-matrix (sans scattering)
    wavelengths_nm : array
        Longueurs d'onde (nm)
    S_eva_cm : array
        Coefficient de scattering EVA (cm^-1) pour chaque lambda
    S_tedlar_cm : array
        Coefficient de scattering Tedlar (cm^-1) pour chaque lambda
    d_eva_cm : float
        Epaisseur EVA (cm)
    d_tedlar_cm : float
        Epaisseur Tedlar (cm)
    R_back : float
        Reflectance arriere (Al)

    Returns
    -------
    A_Si_corrected : array
        Absorption Si corrigee pour le scattering
    """
    n_wl = len(wavelengths_nm)
    A_Si_corrected = np.zeros(n_wl)

    for i in range(n_wl):
        A_Si_0 = A_Si_coherent[i]

        # Fraction diffusee par l'EVA
        f_scat_eva = 1.0 - np.exp(-S_eva_cm[i] * d_eva_cm)

        # Fraction diffusee par le Tedlar
        f_scat_tedlar = 1.0 - np.exp(-S_tedlar_cm[i] * d_tedlar_cm)

        # Le scattering dans l'EVA augmente le chemin optique moyen
        # Facteur d'augmentation du chemin optique: ~1/(1-f_scat) pour isotrope
        if f_scat_eva < 0.9:
            path_enhancement = 1.0 + 0.5 * f_scat_eva / (1.0 - f_scat_eva + 0.1)
        else:
            path_enhancement = 2.0

        # Le Tedlar blanc augmente la reflexion arriere effective
        R_back_eff = R_back + (1.0 - R_back) * f_scat_tedlar * 0.9  # 90% retrodiffuse

        # Correction de A_Si pour le light trapping ameliore
        # Plus de passes = plus d'absorption
        # A_Si_corr = A_Si_0 * path_enhancement * (1 + R_back_eff * (1-A_Si_0))
        extra_absorption = (1.0 - A_Si_0) * R_back_eff * A_Si_0 * path_enhancement * 0.5

        A_Si_corrected[i] = min(1.0, A_Si_0 + extra_absorption)

    return A_Si_corrected


class VolumetricScatteringModel:
    """
    Modele de scattering volumique pour modules PV.

    Combine:
    - Kubelka-Munk pour les couches diffusantes
    - Correction du light-trapping
    - Coefficients de scattering calibrables
    """

    def __init__(self,
                 d_eva_um: float = 450.0,
                 d_tedlar_um: float = 350.0,
                 S_eva_base: float = 0.0,
                 S_tedlar_base: float = 50.0,
                 tio2_concentration: float = 0.10,
                 eva_bubble_density: float = 0.0,
                 R_back: float = 0.85):
        """
        Parameters
        ----------
        d_eva_um : float
            Epaisseur EVA (um)
        d_tedlar_um : float
            Epaisseur Tedlar (um)
        S_eva_base : float
            Coefficient de scattering de base EVA (cm^-1)
        S_tedlar_base : float
            Coefficient de scattering de base Tedlar (cm^-1)
        tio2_concentration : float
            Concentration TiO2 dans Tedlar (fraction volumique)
        eva_bubble_density : float
            Densite de bulles dans EVA (cm^-3)
        R_back : float
            Reflectance arriere Al
        """
        self.d_eva_cm = d_eva_um * 1e-4
        self.d_tedlar_cm = d_tedlar_um * 1e-4
        self.S_eva_base = S_eva_base
        self.S_tedlar_base = S_tedlar_base
        self.tio2_concentration = tio2_concentration
        self.eva_bubble_density = eva_bubble_density
        self.R_back = R_back

    def get_scattering_coefficients(self, wavelengths_nm: np.ndarray
                                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule les coefficients de scattering spectraux.

        Returns
        -------
        S_eva : array
            Coefficients de scattering EVA (cm^-1)
        S_tedlar : array
            Coefficients de scattering Tedlar (cm^-1)
        """
        n_wl = len(wavelengths_nm)
        S_eva = np.zeros(n_wl)
        S_tedlar = np.zeros(n_wl)

        for i, wl in enumerate(wavelengths_nm):
            S_eva[i] = self.S_eva_base + get_eva_scattering_coefficient(
                wl, self.eva_bubble_density, 0.0
            )
            S_tedlar[i] = self.S_tedlar_base + get_tedlar_scattering_coefficient(
                wl, self.tio2_concentration
            )

        return S_eva, S_tedlar

    def apply_correction(self, A_Si_coherent: np.ndarray,
                          wavelengths_nm: np.ndarray) -> np.ndarray:
        """
        Applique la correction de scattering volumique.

        Parameters
        ----------
        A_Si_coherent : array
            Absorption Si depuis S-matrix
        wavelengths_nm : array
            Longueurs d'onde

        Returns
        -------
        A_Si_corrected : array
            Absorption corrigee
        """
        S_eva, S_tedlar = self.get_scattering_coefficients(wavelengths_nm)

        return apply_volumetric_scattering_nb(
            A_Si_coherent, wavelengths_nm,
            S_eva, S_tedlar,
            self.d_eva_cm, self.d_tedlar_cm,
            self.R_back
        )


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    wavelengths = np.linspace(300, 1200, 100)

    # Test scattering coefficients
    S_eva = np.array([get_eva_scattering_coefficient(wl, bubble_density=1e5)
                      for wl in wavelengths])
    S_tedlar = np.array([get_tedlar_scattering_coefficient(wl, tio2_concentration=0.10)
                         for wl in wavelengths])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(wavelengths, S_eva, 'b-', lw=2, label='EVA (bulles)')
    ax.set_xlabel('Longueur d\'onde (nm)')
    ax.set_ylabel('S (cm$^{-1}$)')
    ax.set_title('Scattering EVA')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(wavelengths, S_tedlar, 'r-', lw=2, label='Tedlar (TiO2 10%)')
    ax.set_xlabel('Longueur d\'onde (nm)')
    ax.set_ylabel('S (cm$^{-1}$)')
    ax.set_title('Scattering Tedlar')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = PROJECT_ROOT / 'results' / 'scattering_coefficients.png'
    plt.savefig(fig_path, dpi=150)
    print(f"Figure: {fig_path}")
    plt.close()

    # Test Kubelka-Munk
    print("\nTest Kubelka-Munk pour Tedlar blanc:")
    K = 0.1  # cm^-1 (faible absorption)
    S = 100.0  # cm^-1 (fort scattering)
    d = 0.035  # 350 um

    R, T = kubelka_munk_RT(K, S, d)
    print(f"  K={K} cm^-1, S={S} cm^-1, d={d*1e4} um")
    print(f"  R={R:.3f}, T={T:.3f}, A={1-R-T:.3f}")
