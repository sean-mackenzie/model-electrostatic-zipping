from os.path import join
import numpy as np
import matplotlib.pyplot as plt

from utils.shapes import radius_of_curvature_and_curvature_from_angle_and_arc_length, annular_area
from utils.energy import bending_energy_density, mechanical_energy_density_NeoHookean


def plot_bending_to_stretching_energy_density_ratio_by_bending_angle(
        bending_angles, bending_length_scale, pre_stretches, E, nu, t, path_save=None, save_id=None,):
    # iterate and plot
    fig, ax = plt.subplots(figsize=(4.5, 3.75))

    for bending_angle_degrees in bending_angles:
        R, k = radius_of_curvature_and_curvature_from_angle_and_arc_length(
            angle_degrees=bending_angle_degrees, arc_length=bending_length_scale)
        U_bending = bending_energy_density(E=E, nu=nu, t=t, curvature=k)

        U_S = []
        U_B2S = []
        for pre_stretch in pre_stretches:
            U_stretching = mechanical_energy_density_NeoHookean(mu=E / 3, l=pre_stretch)
            U_B2S.append(U_bending / U_stretching)
            U_S.append(U_stretching)

        ax.plot(pre_stretches, U_B2S, lw=1, label=np.round(bending_angle_degrees, 1))
    # ax.plot(pre_stretches, U_S, color='tab:red', lw=1.25, label=r'$U_{stretching}$')
    ax.set_xlabel('Pre-stretch')
    ax.set_ylabel(r'$U_{bending}/U_{stretching}$')
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(alpha=0.25)
    ax.legend(title=r'$\theta_{bend} \: (\degree)$')
    ax.set_title(r'$L_{bend} = $' + f'{bending_length_scale * 1e6:.1f} $\mu$m')
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, save_id + 'U_bending-to-stretching_by_angle.png'),
                    dpi=300, facecolor='w', bbox_inches='tight')
    plt.show()
    plt.close()

def plot_bending_to_stretching_energy_density_ratio_by_bending_length_scale(
        bending_length_scales, bending_angle, pre_stretches, E, nu, t, path_save=None, save_id=None,):
    # iterate and plot
    fig, ax = plt.subplots(figsize=(4.5, 3.75))

    for bending_length_scale in bending_length_scales:
        R, k = radius_of_curvature_and_curvature_from_angle_and_arc_length(
            angle_degrees=bending_angle, arc_length=bending_length_scale)
        U_bending = bending_energy_density(E=E, nu=nu, t=t, curvature=k)

        U_S = []
        U_B2S = []
        for pre_stretch in pre_stretches:
            U_stretching = mechanical_energy_density_NeoHookean(mu=E / 3, l=pre_stretch)
            U_B2S.append(U_bending / U_stretching)
            U_S.append(U_stretching)

        ax.plot(pre_stretches, U_B2S, lw=1, label=np.round(bending_length_scale * 1e6, 1))
    # ax.plot(pre_stretches, U_S, color='tab:red', lw=1.25, label=r'$U_{stretching}$')
    ax.set_xlabel('Pre-stretch')
    ax.set_ylabel(r'$U_{bending}/U_{stretching}$')
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(alpha=0.25)
    ax.legend(title=r'$L_{bend} \: (\mu m)$')
    ax.set_title(r'$\theta_{bend} = $' + f'{bending_angle:.1f} $\degree$')
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, save_id + 'U_bending-to-stretching_by_length.png'),
                    dpi=300, facecolor='w', bbox_inches='tight')
    plt.show()
    plt.close()

def bending_energy_per_unit_area(E, nu, t, curvature):
    """
    Computes the bending energy per unit area for a given material and geometric configuration.
    The calculation is based on the material's elastic modulus, Poisson's ratio, thickness,
    and the curvature of the surface under deformation.

    If necessary, a good approximation for curvature is: k ~ 1 / L, where L is length scale of deformation.
    U_bending = D * curvature ** 2, which scales as: U_bending ~ E * t ** 3 / L^2
    """
    D = E * t ** 3 / (12 * (1 - nu ** 2))  # Bending stiffness
    U_bending = D * curvature ** 2  # per unit area
    return U_bending

def stretching_energy_per_unit_area(E, t, stretch):
    """
    Stretching energy per unit area: U_stretching ~ T * epsilon **2
    where:
        epsilon is in-plane strain (strain = stretch - 1)
        T ~ E * t is the in-plane membrane tension (or more precisely, T = E * t * epsilon).
            So, if you want to do characteristic scaling, use T ~ E * t; or,
            if you want accuracy, use T = E * t * epsilon.
    U_stretching then scales as: U_stretching ~ E * t * epsilon ** 2
    """
    epsilon = stretch - 1
    U_stretching = E * t * epsilon ** 3
    return U_stretching

def bending_energy_per_unit_area_characteristic_scaling(E, t, curvature):
    """
    Computes the bending energy per unit area for a given material and geometric configuration.
    The calculation is based on the material's elastic modulus, Poisson's ratio, thickness,
    and the curvature of the surface under deformation.

    If necessary, a good approximation for curvature is: k ~ 1 / L, where L is length scale of deformation.
    U_bending = D * curvature ** 2, which scales as: U_bending ~ E * t ** 3 / L^2
    """
    U_bending_scaling = E * t ** 3 * curvature ** 2  # per unit area
    return U_bending_scaling

def stretching_energy_per_unit_area_characteristic_scaling(E, t, stretch):
    """
    Stretching energy per unit area: U_stretching ~ T * epsilon **2
    where:
        epsilon is in-plane strain (strain = stretch - 1)
        T ~ E * t is the in-plane membrane tension (or more precisely, T = E * t * epsilon).
            So, if you want to do characteristic scaling, use T ~ E * t; or,
            if you want accuracy, use T = E * t * epsilon.
    U_stretching then scales as: U_stretching ~ E * t * epsilon ** 2
    """
    epsilon = stretch - 1
    U_stretching = E * t * epsilon ** 2
    return U_stretching

def dimensionless_bending_stretching_ratio_characteristic_scaling(t, stretch, curvature):
    """
    t: thickness of membrane (m)
    stretch: stretching factor (1 + stretch)
    curvature: radius of curvature (a good approximation is 1 / L, where L is length scale of deformation)

    U_bending ~ E * t ** 3 / L^2
    U_stretching ~ E * t * epsilon ** 2
    U_bending / U_stretching = (t / L) ** 2 * (1 / epsilon) ** 2

    Physical interpretation:
        * (t / L) ** 2 --> a geometric factor, stating that thinner membranes or more gradual curvature reduces relevance of bending
        * (1 / epsilon) ** 2 --> mechanical factor, stating that more stretching makes bending less relevant

    A key takeaway is that bending energy scales with t ** 3, while stretching scales with t.
    So, the relevance of bending is largely tied to the thickness of the membrane.
    """
    epsilon = stretch - 1
    L = 1 / curvature
    U_bending_to_stretching_ratio = (t / L) ** 2 * (1 / epsilon) ** 2
    return U_bending_to_stretching_ratio


if __name__ == '__main__':
    # -
    SAVE_DIR = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/thesis/zipper_actuator/Important notes/bending-vs-stretching'
    SAVE_ID = 'representative'

    # 1. Inputs
    # Geometry
    D = 4e-3  # Diameter (e.g., W18, from early zipping)
    t = 200e-6  # Membrane thickness (e.g, 200 um SILPURAN membrane)
    # Mechanical properties
    E_comp = 5e6  # Young's modulus of composite membrane
    nu = 0.499
    pre_stretch = 1.00001

    # 1b. Define assumed bending
    bending_angle_degrees = 1  # angle of bending (e.g., sidewall slope angle)
    # bending length scale: the membrane bends around this angle
    # over a length span of 2X the bending length scale
    # (1X on each side of angular bend)
    bending_length_scale = D / 2

    # 1c. Sweep parameters
    # radius_m: the radius of the surface profile (at the location of bending)
    radii = np.linspace(0.5, 2, 250) * 1e-3  # radius of the circular annulus (m)
    pre_stretches = np.linspace(1.00001, 1.2, 200)
    bending_angles = np.linspace(1, 30, 6)  # sidewall slope angle (degrees)
    bending_length_scales = np.linspace(D / 10, D / 2, 6)  # characteristic length of bending (m)

    # ---
    plate_aspect_ratio = t / D
    if plate_aspect_ratio > 0.01:
        print(f"This plate exceeds the typical aspect ratio ({plate_aspect_ratio}) of t/D < 0.01 for membranes. ")
    else:
        print(f"t/D = {plate_aspect_ratio}, within typical t/D < 0.01 for membranes. ")

    U_bending = bending_energy_per_unit_area(E=E_comp, nu=nu, t=t, curvature=bending_length_scale)
    U_stretching = stretching_energy_per_unit_area(E=E_comp, t=t, stretch=pre_stretch)
    U_bending_to_stretching_ratio = dimensionless_bending_stretching_ratio_characteristic_scaling(
        t=t, stretch=pre_stretch, curvature=bending_length_scale,)
    # -
    print(f"Bending energy density: {U_bending:.3e} J/m^2")
    print(f"Stretching energy density: {U_stretching:.3e} J/m^2")
    print(f"Ratio of bending energy density to stretching energy density: {U_bending / U_stretching:.3e}")
    print(f"Ratio calc: {U_bending / U_stretching:.3e}")

    # 2. Calculate the radius of curvature
    R, k = radius_of_curvature_and_curvature_from_angle_and_arc_length(
        angle_degrees=bending_angle_degrees, arc_length=bending_length_scale)

    print(f"Radius of curvature: {R * 1e6:.3f} um")
    print(f"Curvature: {k:.3f} 1/m")

    # 3. Calculate bending energy density (bending energy per unit area)
    U_bending = bending_energy_density(E=E_comp, nu=nu, t=t, curvature=k)
    print(f"Bending energy density: {U_bending:.3e} J/m^2")

    # 4. Calculate the stretching energy density (use typical Gent or Neo-Hookean model)
    U_stretching = mechanical_energy_density_NeoHookean(mu=E_comp / 3, l=pre_stretch)
    print(f"Stretching energy density: {U_stretching:.3e} J/m^2")

    # 5. Ratio of bending energy density to stretching energy density
    U_bending_to_stretching_ratio = U_bending / U_stretching
    print(f"Ratio of bending energy density to stretching energy density: {U_bending_to_stretching_ratio:.3e}")

    # calculate the bending energy (units: J) from bending energy density (units: J/m^2)
    # radius_m: the radius of the surface profile (at the location of bending)
    # width_m: the width of the cicular annulus (here, this is the bending length scale)
    bending_energy = U_bending * annular_area(radius_m=radii, width_m=bending_length_scale)
    fig, ax = plt.subplots(figsize=(4.5, 3.75))
    ax.plot(radii * 1e3, bending_energy * 1e9, lw=1)
    ax.set_xlabel('Radius (mm)')
    ax.set_ylabel('Bending energy (nJ)')
    ax.set_title(r'$L_{bend} = $' + f'{bending_length_scale * 1e6:.1f} $\mu$m;' +
                  r'$\theta_{bend} = $' + f'{bending_angle_degrees:.1f} $\degree$')
    ax.grid(alpha=0.25)
    plt.suptitle(f'Plate: D = {D * 1e3:.1f} mm, t = {t * 1e6:.1f} um, E = {E_comp * 1e-6:.1f} MPa')
    plt.tight_layout()
    plt.savefig(join(SAVE_DIR, SAVE_ID + '_bending_energy.png'), dpi=300, facecolor='w', bbox_inches='tight')
    plt.show()

    # ----

    # Toy example

    # vary bending angle
    # bending_length_scale = 40e-6  # the membrane bends over this length (meters)
    plot_bending_to_stretching_energy_density_ratio_by_bending_angle(
            bending_angles, bending_length_scale, pre_stretches, E_comp, nu, t,
        path_save=SAVE_DIR, save_id=SAVE_ID,)

    # vary bending length scale
    # bending_angle = 10
    plot_bending_to_stretching_energy_density_ratio_by_bending_length_scale(
            bending_length_scales, bending_angle_degrees, pre_stretches, E_comp, nu, t,
        path_save=SAVE_DIR, save_id=SAVE_ID,)
