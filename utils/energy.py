# tests/test_model_sweep.py

# imports
import numpy as np

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