# tests/test_model_sweep.py

# imports
import numpy as np
from scipy.optimize import fsolve

# constants
eps0 = 8.854e-12

# functions: energy
def mechanical_energy_density_Gent(mu, J, l):
    return -1 * (mu * J / 2) * np.log(1 - ((2 * l ** 2 + 1 / l ** 4 - 3) / J))

def capacitance_surface_roughness(eps_r, A, d, R0):
    return eps0 * A / R0 * np.log(eps_r * R0 / d + 1)

def electrostatic_energy_density_SR(eps_r, A, d, U, R0):
    if R0 == 0:
        R0 = 1e-9
    return -0.5 * U ** 2 * capacitance_surface_roughness(eps_r, A, d, R0)

# --- The following is a reformulation and should be double-checked

# Define the Gent model stress function for equi-biaxial stretch
def gent_stress_from_pre_stretch(pre_stretch, mu, Jm):
    """
    Computes the stress (true stress?) in a material using the Gent hyperelastic model
    based on the given pre_stretch value, material's shear modulus,
    and limiting chain extensibility parameter. This function calculates
    the first invariant of the deformation tensor, then computes the
    numerator and denominator for the stress formula, and finally
    calculates the stress in the material.

    :param pre_stretch: Stretch ratio of the material
    :type pre_stretch: float
    :param mu: Shear modulus of the material (Shear modulus = Young's modulus / 3)
    :type mu: float
    :param Jm: Limiting extensibility parameter of the material
    :type Jm: float
    :return: Computed stress of the material as per the Gent model
    :rtype: float
    """
    I1 = 2 * pre_stretch**2 + 1 / pre_stretch**4
    numerator = pre_stretch**2 - 1 / pre_stretch**4
    denominator = 1 - (I1 - 3) / Jm
    stress = mu * numerator / denominator
    return stress


def neo_hookean_stress_from_pre_stretch(pre_stretch, shear_modulus):
    """
    Computes the Cauchy stress (true stress) for equibiaxial deformation based on the given pre-stretch
    and the material's shear modulus.

    This function calculates the true stress for materials undergoing equibiaxial deformation
    using a derived formula. The input parameters include the pre-stretch ratio and the
    shear modulus of the material. It is recommended to verify the inputs for accuracy
    before using this function in a computational context. The mathematical formulation
    assumes ideal conditions without considering additional complexities such as temperature
    or anisotropic effects in materials.

    :param pre_stretch: The pre-stretch ratio of the material used in deformation calculations.
        It must be a positive float value, indicating the ratio by which the material is
        stretched from its original length.
    :type pre_stretch: float

    :param shear_modulus: The shear modulus of the material. Represents the material's
        stiffness in response to shearing deformation.
    :type shear_modulus: float

    :return: The computed Cauchy true stress for the defined equibiaxial pre-stretch.
    :rtype: float
    """
    true_stress = shear_modulus * (pre_stretch**2 - 1 / pre_stretch**4)
    return true_stress


# Define function to compute stretch from stress using Gent model (numerical root-finding)
def compute_gent_stretch_from_stress(sigma, mu, Jm, initial_guess=1.5):
    def gent_equation(lmbda):
        I1 = 2 * lmbda ** 2 + 1 / lmbda ** 4
        numerator = lmbda ** 2 - 1 / lmbda ** 4
        denominator = 1 - (I1 - 3) / Jm
        return mu * numerator / denominator - sigma

    solution = fsolve(gent_equation, x0=initial_guess)
    return solution[0]


# Define a function to solve for lambda given sigma and mu
def compute_neo_hookean_stretch_from_stress(sigma, mu, initial_guess=1.5):
    # Define the equation to solve: lambda^6 - (sigma/mu)*lambda^4 - 1 = 0
    def stretch_equation(lmbda, sigma, mu):
        return lmbda ** 6 - (sigma / mu) * lmbda ** 4 - 1

    solution = fsolve(stretch_equation, x0=initial_guess, args=(sigma, mu))
    return solution[0]