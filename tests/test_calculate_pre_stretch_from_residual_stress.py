import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

from utils import energy

if __name__ == '__main__':

    # Material parameters of pristine material (i.e., before pre-stretching)
    E = 1000  # (kPa) Young's modulus
    J_M = 80  # extensibility constant for Gent hyperelastic model

    KNOWN_PRE_STRETCH = 1.252
    KNOWN_SIGMA_0 = 345.5  # (kPa) residual stress

    # --- ---

    shear_modulus = E / 3

    # 1. calculate the residual stress due to pre-stretching (pre-stretch must be known)
    gent_stress_due_to_pre_stretching = energy.gent_stress_from_pre_stretch(KNOWN_PRE_STRETCH, shear_modulus, J_M)
    neo_hookean_stress_due_to_pre_stretching = energy.neo_hookean_stress_from_pre_stretch(KNOWN_PRE_STRETCH, shear_modulus)
    print("Gent residual stress:", gent_stress_due_to_pre_stretching)
    print("Neo-Hookean residual stress:", neo_hookean_stress_due_to_pre_stretching)

    # ---

    # 2. (NOTE: only valid for non-metallized membrane) calculate pre-stretch that gives rise to measured stress
    gent_pre_stretch = energy.compute_gent_stretch_from_stress(
        sigma=KNOWN_SIGMA_0,
        mu=shear_modulus,
        Jm=J_M,
        initial_guess=1.5,
    )
    neo_hookean_pre_stretch = energy.compute_neo_hookean_stretch_from_stress(
        sigma=KNOWN_SIGMA_0,
        mu=shear_modulus,
        initial_guess=1.5,
    )
    print("Gent pre-stretch:", gent_pre_stretch)
    print("Neo-Hookean pre-stretch:", neo_hookean_pre_stretch)