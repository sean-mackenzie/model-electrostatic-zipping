

def calculate_biaxial_strain_and_stretch(stress, youngs_modulus, poisson_ratio):
    """
    Calculate the biaxial strain and stretch for a metal film under biaxial stress.

    Parameters:
        stress (float): The residual stress in the film (Pa).
        youngs_modulus (float): Young's modulus of the material (Pa).
        poisson_ratio (float): Poisson's ratio of the material (dimensionless).

    Returns:
        tuple: A tuple containing:
            - biaxial_strain (float): The computed biaxial strain (dimensionless).
            - biaxial_stretch (float): The computed biaxial stretch (dimensionless).
    """
    # Calculate the effective biaxial modulus
    biaxial_modulus = youngs_modulus / (1 - poisson_ratio)

    # Calculate the biaxial strain
    biaxial_strain = stress / biaxial_modulus

    # Calculate the biaxial stretch
    biaxial_stretch = 1 + biaxial_strain

    return biaxial_strain, biaxial_stretch


if __name__ == '__main__':
    # Example usage
    stress = 40e3  # Residual stress in Pa (40 kPa)
    youngs_modulus = 5e9  # Young's modulus in Pa (~20nm Au on MPTMS on ELASTOSIL)
    poisson_ratio = 0.44  # Poisson's ratio (dimensionless)

    strain, stretch = calculate_biaxial_strain_and_stretch(stress, youngs_modulus, poisson_ratio)
    print(f"Biaxial Strain: {strain:.6f}")
    print(f"Biaxial Stretch: {stretch:.6f}")

    import numpy as np
    import matplotlib.pyplot as plt

    Es = np.array([2.5, 5, 10, 25, 50]) * 1e9
    sigma0s = np.array([30, 45, 75, 100, 150, 200]) * 1e3

    fig, ax = plt.subplots(figsize=(8, 5))
    for E in Es:
        strains, stretchs = calculate_biaxial_strain_and_stretch(sigma0s, E, poisson_ratio)
        ax.plot(sigma0s, stretchs, label=f"E = {E * 1e-9} GPa")
    ax.set_xlabel("Residual Stress (Pa)")
    ax.set_ylabel(r'$\lambda$')
    ax.set_title("Biaxial Stretch vs. Residual Stress for Au film")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()