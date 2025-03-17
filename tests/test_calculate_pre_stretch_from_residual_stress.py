import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

from utils import energy

if __name__ == '__main__':
    E = 3e3  # (kPa) Young's modulus
    SIGMA_0 = 30  # (kPa) residual stress
    J_M = 80  # extensibility constant for Gent hyperelastic model

    # --- compute
    shear_modulus = E / 3
    gent_pre_stretch = energy.compute_gent_stretch_from_stress(
        sigma=SIGMA_0,
        mu=shear_modulus,
        Jm=J_M,
        initial_guess=1.5,
    )

    neo_hookean_pre_stretch = energy.compute_neo_hookean_stretch_from_stress(
        sigma=SIGMA_0,
        mu=shear_modulus,
        initial_guess=1.5,
    )

    print("Gent pre-stretch:", gent_pre_stretch)
    print("Neo-Hookean pre-stretch:", neo_hookean_pre_stretch)