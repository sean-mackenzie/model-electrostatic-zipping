import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

""" The following uses a Neo-Hookean model to calculate equibiaxial Cauchy stress """
# Define the equation to solve: lambda^6 - (sigma/mu)*lambda^4 - 1 = 0
def stretch_equation(lmbda, sigma, mu):
    return lmbda**6 - (sigma/mu)*lmbda**4 - 1

# Define a function to solve for lambda given sigma and mu
def compute_stretch_from_stress(sigma, mu, initial_guess=1.5):
    solution = fsolve(stretch_equation, x0=initial_guess, args=(sigma, mu))
    return solution[0]

# Example values
sigma_example = 250  # kPa
mu_example = 333     # kPa
# Solve for lambda
lambda_result = compute_stretch_from_stress(sigma_example, mu_example)
# Display result
lambda_result

# --- Generate a plot

# Define a range of stress values
stress_range = np.linspace(0, 700, 200)  # kPa
# Compute corresponding stretch values using the inverse model
stretch_from_stress = [compute_stretch_from_stress(sigma, mu_example) for sigma in stress_range]
# Plotting
plt.figure(figsize=(8, 5))
plt.plot(stress_range, stretch_from_stress, linewidth=2)
plt.title("Stretch Ratio vs. Cauchy Stress (Inverse Mapping, μ=333 kPa)")
plt.xlabel("Cauchy Stress (kPa)")
plt.ylabel("Stretch Ratio (λ)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----

# ----

def cauchy_stress_from_equibiaxial_pre_stretch(pre_stretch, shear_modulus):
    """ uses Neo-Hookean model to calculate equibiaxial Cauchy stress """
    true_stress = shear_modulus * (pre_stretch**2 - 1 / pre_stretch**4)
    return true_stress

lambda_values = np.linspace(1, 1.5, 100)
cauchy_stress_values = cauchy_stress_from_equibiaxial_pre_stretch(lambda_values, mu_example)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(lambda_values, cauchy_stress_values, linewidth=2)
plt.title("Equi-biaxial Cauchy Stress vs. Stretch (Neo-Hookean, μ=333 kPa)")
plt.xlabel("Stretch Ratio (λ)")
plt.ylabel("Cauchy Stress (kPa)")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------------------------
""" The following include a Gent hyperelastic model"""

# Define the Gent model stress function for equi-biaxial stretch
def gent_stress(lambda_vals, mu, Jm):
    I1 = 2 * lambda_vals**2 + 1 / lambda_vals**4
    numerator = lambda_vals**2 - 1 / lambda_vals**4
    denominator = 1 - (I1 - 3) / Jm
    stress = mu * numerator / denominator
    return stress

# Define parameters
Jm = 80  # Gent parameter
lambda_values = np.linspace(1.0, 3, 300)

# Calculate stress using Gent model
gent_stress_values = gent_stress(lambda_values, mu_example, Jm)

# Also calculate Neo-Hookean for comparison
neo_hookean_stress_values = mu_example * (lambda_values**2 - 1 / lambda_values**4)

# Plotting both
plt.figure(figsize=(8, 5))
plt.plot(lambda_values, gent_stress_values, label='Gent Model (Jm=80)', linewidth=2)
plt.plot(lambda_values, neo_hookean_stress_values, label='Neo-Hookean', linestyle='--', linewidth=2)
plt.title("Equi-biaxial Cauchy Stress vs. Stretch (μ=333 kPa)")
plt.xlabel("Stretch Ratio (λ)")
plt.ylabel("Cauchy Stress (kPa)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Define function to compute stretch from stress using Gent model (numerical root-finding)
def compute_gent_stretch_from_stress(sigma, mu, Jm, initial_guess=1.5):
    def gent_equation(lmbda):
        I1 = 2 * lmbda ** 2 + 1 / lmbda ** 4
        numerator = lmbda ** 2 - 1 / lmbda ** 4
        denominator = 1 - (I1 - 3) / Jm
        return mu * numerator / denominator - sigma

    solution = fsolve(gent_equation, x0=initial_guess)
    return solution[0]


# Generate a range of stress values
stress_range = np.linspace(0, 700, 200)

# Compute corresponding stretch values using Gent model
gent_stretch_from_stress = [compute_gent_stretch_from_stress(sigma, mu_example, Jm) for sigma in stress_range]

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(stress_range, gent_stretch_from_stress, label="Gent Model (Jm=80)", linewidth=2)
plt.title("Stretch Ratio vs. Cauchy Stress (Inverse Mapping, Gent Model, μ=333 kPa, Jm=80)")
plt.xlabel("Cauchy Stress (kPa)")
plt.ylabel("Stretch Ratio (λ)")
plt.grid(True)
plt.tight_layout()
plt.show()


