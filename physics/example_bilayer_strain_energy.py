# Re-run after environment reset

import numpy as np
import matplotlib.pyplot as plt

def bilayer_strain_energy(lambda_vals, mu_elastomer, t_elastomer, E_metal, nu_metal, t_metal, curvature):
    """
    Computes total strain + bending energy per unit area for a metal-elastomer bilayer.

    Parameters:
    - lambda_vals: array of stretch ratios (equi-biaxial)
    - mu_elastomer: shear modulus of elastomer (Pa)
    - t_elastomer: thickness of elastomer (m)
    - E_metal: Young's modulus of metal (Pa)
    - nu_metal: Poisson's ratio of metal
    - t_metal: thickness of metal (m)
    - curvature: scalar curvature (1/radius) in 1D (assumed same in x and y)

    Returns:
    - U_total: total energy per unit area (J/m²)
    """
    # Strain energy in elastomer (neo-Hookean, incompressible)
    I1 = 2 * lambda_vals**2 + 1 / lambda_vals**4
    W_elastomer = 0.5 * mu_elastomer * (I1 - 3)
    U_elastomer = W_elastomer * t_elastomer

    # Strain energy in metal (linear elasticity)
    strain_metal = lambda_vals - 1
    W_metal = (E_metal / (1 - nu_metal)) * strain_metal**2
    U_metal = W_metal * t_metal

    # Bending energy in metal (only)
    D_metal = E_metal * t_metal**3 / (12 * (1 - nu_metal**2))  # Bending stiffness
    U_bending = D_metal * curvature**2  # per unit area

    # Total energy per unit area
    U_total = U_elastomer + U_metal + U_bending
    return U_total

# Example parameters
lambda_range = np.linspace(0.9, 1.5, 200)
mu_elastomer = 333e3           # shear modulus of elastomer (Pa)
t_elastomer = 20e-6            # 20 µm
E_metal = 80e9                 # Young's modulus of gold (Pa)
nu_metal = 0.44                # Poisson's ratio of gold
t_metal = 20e-9                # 20 nm
curvature = 1 / (1e-3)         # curvature = 1 mm⁻¹

# Compute total strain + bending energy
U = bilayer_strain_energy(lambda_range, mu_elastomer, t_elastomer, E_metal, nu_metal, t_metal, curvature)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(lambda_range, U, linewidth=2)
plt.xlabel("Stretch Ratio (λ)")
plt.ylabel("Total Strain + Bending Energy (J/m²)")
plt.title("Bilayer Energy vs. Stretch Ratio (with Bending)")
plt.grid(True)
plt.tight_layout()
plt.show()
