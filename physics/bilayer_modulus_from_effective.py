def estimate_gold_modulus_from_effective(
    E_eff_measured,
    t_elastomer,
    E_elastomer,
    t_gold
):
    """
    Estimates the Young's modulus of the gold film using a rule-of-mixtures model for in-plane membrane stiffness.

    Parameters:
    - E_eff_measured: effective Young's modulus of the bilayer system (Pa)
    - t_elastomer: thickness of the elastomer layer (m)
    - E_elastomer: Young's modulus of the elastomer (Pa)
    - t_gold: thickness of the gold film (m)

    Returns:
    - E_gold_estimated: estimated Young's modulus of the gold film (Pa)
    """
    t_total = t_elastomer + t_gold
    numerator = E_eff_measured * t_total - E_elastomer * t_elastomer
    E_gold_estimated = numerator / t_gold
    return E_gold_estimated

def effective_youngs_modulus_bilayer(t_silicone, E_silicone, t_gold, E_gold):
    """
    Computes the effective Young's modulus of a bilayer system (gold on silicone)
    using a rule-of-mixtures type approximation for plane-stress membrane behavior.

    *NOTE: this is the same equation used by Niklaus and Shea (2010) Electrical conductivity
    and Young's modulus of flexible nanocomposites made by metal-ion implantation...

    Parameters:
    - t_silicone: thickness of the silicone layer (in meters)
    - E_silicone: Young's modulus of silicone (in Pascals)
    - t_gold: thickness of the gold film (in meters)
    - E_gold: Young's modulus of gold (in Pascals)

    Returns:
    - E_effective: Effective Young's modulus of the bilayer (in Pascals)
    """

    total_thickness = t_silicone + t_gold
    E_effective = (E_silicone * t_silicone + E_gold * t_gold) / total_thickness
    return E_effective


if __name__ == '__main__':

    # Example data from: 20250409_CX23-0pT-20nmAu_4mmDia
    E_eff_measured = 3.81e6     # Pa (measured from bulge test)
    t_elastomer = 20e-6         # m
    E_elastomer = 1.1e6         # Pa
    t_gold = 20e-9              # m

    E_gold_estimated = estimate_gold_modulus_from_effective(
        E_eff_measured,
        t_elastomer,
        E_elastomer,
        t_gold
    )

    print(f'Estimated gold modulus: {E_gold_estimated:.2e} Pa')

    import numpy as np
    import matplotlib.pyplot as plt

    t_gold = np.linspace(0, 35) * 1e-9
    E_effective = effective_youngs_modulus_bilayer(t_elastomer, E_elastomer, t_gold, E_gold_estimated)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_gold * 1e9, E_effective * 1e-6, linewidth=2)
    ax.set_xlabel("Au Thickness (nm)")
    ax.set_ylabel(r"$E_{eff} \: (MPa)$")
    ax.set_title("Effective Young's Modulus of Bilayer vs. Au Thickness")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()
