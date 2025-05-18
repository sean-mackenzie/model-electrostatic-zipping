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


if __name__ == '__main__':
    # -
    SAVE_DIR = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/zipper_paper/Testing/ref/misc_figs'
    SAVE_ID = 'representative'
    # 1. Inputs
    bending_angle_degrees = 15  # sidewall slope angle
    bending_length_scale = 20e-6  # the membrane bends around this 10 degree angle
    # over a length span of ~80 um in total, 40 before and 40 after the bend.
    E_comp = 5e6  # Young's modulus of composite membrane
    nu = 0.499
    t = 20e-6
    pre_stretch = 1.001

    # 2. Calculate the radius of curvature
    R, k = radius_of_curvature_and_curvature_from_angle_and_arc_length(
        angle_degrees=bending_angle_degrees, arc_length=bending_length_scale)

    print(f"Radius of curvature: {R * 1e6:.3f} um")
    print(f"Curvature: {k:.3f} 1/m")

    # 3. Calculate bending energy density (bending energy per unit area)
    U_bending = bending_energy_density(E=E_comp, nu=nu, t=t, curvature=k)
    print(f"Bending energy density: {U_bending:.3f} J/m^2")

    # 4. Calculate the stretching energy density (use typical Gent or Neo-Hookean model)
    U_stretching = mechanical_energy_density_NeoHookean(mu=E_comp / 3, l=pre_stretch)
    print(f"Stretching energy density: {U_stretching:.3f} J/m^2")

    # 5. Ratio of bending energy density to stretching energy density
    U_bending_to_stretching_ratio = U_bending / U_stretching
    print(f"Ratio of bending energy density to stretching energy density: {U_bending_to_stretching_ratio:.5f}")

    # calculate the bending energy (units: J) from bending energy density (units: J/m^2)
    # radius_m: the radius of the surface profile (at the location of bending)
    # width_m: the width of the cicular annulus (here, this is the bending length scale)
    radii = np.linspace(0.5, 2, 250) * 1e-3  # radius of the circular annulus (m)
    bending_energy = U_bending * annular_area(radius_m=radii, width_m=bending_length_scale)
    fig, ax = plt.subplots(figsize=(4.5, 3.75))
    ax.plot(radii * 1e3, bending_energy * 1e9, lw=1)
    ax.set_xlabel('Radius (mm)')
    ax.set_ylabel('Bending energy (nJ)')
    ax.set_title(r'$L_{bend} = $' + f'{bending_length_scale * 1e6:.1f} $\mu$m' + '\n' +
                  r'$\theta_{bend} = $' + f'{bending_angle_degrees:.1f} $\degree$')
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(join(SAVE_DIR, SAVE_ID + '_bending_energy.png'), dpi=300, facecolor='w', bbox_inches='tight')
    plt.show()

    # ----

    # Toy example
    pre_stretches = np.linspace(1.00001, 1.1, 200)

    # vary bending angle
    bending_angles = np.linspace(5, 30, 8)  # sidewall slope angle (degrees)
    bending_length_scale = 40e-6  # the membrane bends over this length (meters)
    plot_bending_to_stretching_energy_density_ratio_by_bending_angle(
            bending_angles, bending_length_scale, pre_stretches, E_comp, nu, t,
        path_save=SAVE_DIR, save_id=SAVE_ID,)

    # vary bending length scale
    bending_length_scales = np.linspace(5, 40, 8) * 1e-6  # sidewall slope angle
    bending_angle = 10
    plot_bending_to_stretching_energy_density_ratio_by_bending_length_scale(
            bending_length_scales, bending_angle, pre_stretches, E_comp, nu, t,
        path_save=SAVE_DIR, save_id=SAVE_ID,)
