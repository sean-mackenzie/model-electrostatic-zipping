# Re-run code after environment reset

import numpy as np
from scipy.integrate import simpson as simps

def compute_bending_energy_small_angle_approximation(r, z, E, nu, t):
    """
    Computes the bending energy of an axisymmetric bilayer profile.
    Small angle approximation: valid for slope angles less than 6-10 degrees.
        principal curvatures are approximated as small-slope lines.

    Parameters:
    - r: 1D numpy array of radial positions (meters)
    - z: 1D numpy array of z-displacement (meters), same length as r
    - E: Young's modulus of the bending layer (Pa)
    - nu: Poisson's ratio of the bending layer
    - t: thickness of the bending layer (meters)

    Returns:
    - U_bend: Total bending energy (Joules)
    - kappa_r: radial curvature array
    - kappa_theta: azimuthal curvature array
    """

    # Bending stiffness
    D = E * t**3 / (12 * (1 - nu**2))

    # First and second derivatives of z with respect to r
    dz_dr = np.gradient(z, r)
    d2z_dr2 = np.gradient(dz_dr, r)

    # Principal curvatures (small-slope approximation)
    kappa_r = d2z_dr2
    kappa_theta = dz_dr / r

    # Handle r=0 singularity (force symmetry: slope = 0 at center)
    kappa_theta[0] = kappa_theta[1]

    # Bending energy density per unit area
    w_bend = 0.5 * D * (kappa_r**2 + 2 * nu * kappa_r * kappa_theta + kappa_theta**2)

    # Total bending energy: integrate 2π * w_bend * r dr
    integrand = 2 * np.pi * w_bend * r
    U_bend = simps(integrand, r)

    return U_bend, kappa_r, kappa_theta


def compute_bending_energy_full_curvature(r, z, E, nu, t):
    """
    Computes the bending energy of an axisymmetric bilayer profile using full curvature expressions.

    Parameters:
    - r: 1D numpy array of radial positions (meters)
    - z: 1D numpy array of z-displacement (meters), same length as r
    - E: Young's modulus of the bending layer (Pa)
    - nu: Poisson's ratio of the bending layer
    - t: thickness of the bending layer (meters)

    Returns:
    - U_bend: Total bending energy (Joules)
    - kappa_r: radial curvature array
    - kappa_theta: azimuthal curvature array
    """

    # Bending stiffness
    D = E * t**3 / (12 * (1 - nu**2))

    # First and second derivatives of z with respect to r
    dz_dr = np.gradient(z, r)
    d2z_dr2 = np.gradient(dz_dr, r)

    # Full (nonlinear) expressions for principal curvatures
    denom = (1 + dz_dr**2)
    kappa_r = d2z_dr2 / (denom**(3/2))
    kappa_theta = (dz_dr / r) / (denom**0.5)

    # Handle r=0 singularity by symmetry assumption
    kappa_theta[0] = kappa_theta[1]

    # Bending energy density per unit area
    w_bend = 0.5 * D * (kappa_r**2 + 2 * nu * kappa_r * kappa_theta + kappa_theta**2)

    # Total bending energy: integrate 2π * w_bend * r dr
    integrand = 2 * np.pi * w_bend * r
    U_bend = simps(integrand, r)

    return U_bend, kappa_r, kappa_theta


if __name__ == '__main__':
    # Example usage with synthetic dome profile
    R = 1.585e-3  # radius in meters
    DZ = 100e-6  # Peak deflection of synthetic dome profile

    # ---

    # Material properties
    # Metal
    E_gold = 80e9       # Pa
    nu_gold = 0.44
    t_gold = 20e-9      # meters
    # Elastomer
    E_elastomer = 10.1e6  # 1 MPa
    nu_elastomer = 0.499
    t_elastomer = 20e-6  # 20 microns

    # ---

    # make r, z values for dome profile
    r_vals = np.linspace(0, R, 300)
    z_vals = DZ * (1 - (r_vals / R)**2)  # simple paraboloid: z = h (1 - (r/R)^2)

    # return total bending energy in Joules
    # using small-angle approximation
    U_bend, kappa_r, kappa_theta = compute_bending_energy_small_angle_approximation(r_vals, z_vals, E_gold, nu_gold, t_gold)

    # return total bending energy in Joules using full curvature expressions
    U_bend_full, kappa_r_full, kappa_theta_full = compute_bending_energy_full_curvature(
        r_vals, z_vals, E_gold, nu_gold, t_gold
    )
    # mean curvature from radially-dependent curvature
    H_m = 0.5 * (kappa_r_full + kappa_theta_full)  # mean curvature in m⁻¹ as a function of radial position
    H_mm = H_m * 1e-3  # mean curvature in mm⁻¹ as a function of radial position
    # Convert r to mm if needed
    r_mm = r_vals * 1e3  # if r is in meters
    numerator = np.trapezoid(H_mm * r_mm, r_mm)
    denominator = np.trapezoid(r_mm, r_mm)
    H_avg_mm = numerator / denominator

    # Compute bending energy for the elastomer using the same curvature profile
    U_bend_elastomer, _, _ = compute_bending_energy_full_curvature(
        r_vals, z_vals, E_elastomer, nu_elastomer, t_elastomer
    )

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(r_vals * 1e3, z_vals * 1e6, label='Mean Curvature: {:.2e} mm^-1'.format(H_avg_mm))
    ax.set_xlabel('Radius (mm)')
    ax.set_ylabel('Deflection (um)')
    ax.legend()
    ax.set_title('Metal (approx, full): ({:.2e}, {:.2e}) J, Elastomer: {:.2e} J'.format(
        U_bend, U_bend_full, U_bend_elastomer))
    plt.suptitle('Bending energy of dome-like deflection profile')
    plt.show()

    # bending stiffness of a circular plate
    t_elastomer = 50e-6
    D = E_elastomer * t_elastomer**3 / (12 * (1 - nu_elastomer**2))
    U = 0.5 * D * (kappa_r**2 + 2 * nu_elastomer * kappa_r * kappa_theta + kappa_theta**2)
    print(U)
    angular_deflection_degrees = 35
    angular_deflection_radians = np.deg2rad(angular_deflection_degrees)
    U2 = 0.5 * D * angular_deflection_radians**2
    print("{} uJ: Bending energy of {}-um thick 'plate' (?) (E={} MPa) with "
          "angular deflection of {} degrees.".format(U2 * 1e6, t_elastomer * 1e6, E_elastomer*1e-6, angular_deflection_degrees))
    # U3 = 0.5 * D * 10**2  # NOTE: this doesn't make sense since angular deflection is degrees
    # print(U3)

    """ NOTE: the below expression is correct for membrane theory """
    # bending stiffness of a circular membrane, assuming constant curvature
    U4 = 0.5 * np.pi * D * (1 + nu_elastomer)**2 * angular_deflection_radians**2
    print("{} uJ: Bending energy of {}-um thick 'membrane'(?) (E={} MPa) with "
          "angular deflection of {} degrees.".format(U4 * 1e6, t_elastomer * 1e6, E_elastomer*1e-6, angular_deflection_degrees))
    a = 1