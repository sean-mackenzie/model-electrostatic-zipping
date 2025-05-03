# test_model_representative_example.py

import os
from os.path import join
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def discretize_profile(r1, z1, r2, z2, r_spacing=1):
    # r1, z1 = -diameter / 2, 0 --> r1 always assumed to be radius
    # r2, z2 = -d0X, -thickness
    extend_flat = np.abs(r1) / 10

    r_left_edge = r1 - extend_flat

    r_left_flat = np.arange(r_left_edge, r1, r_spacing)
    z_left_flat = np.zeros_like(r_left_flat)

    r_slope = np.arange(r1, r2, r_spacing)
    z_slope = np.linspace(z1, z2, len(r_slope))

    if r2 < -2:
        r_bottom_flat = np.arange(r2, 0, r_spacing)
        z_bottom_flat = np.ones_like(r_bottom_flat) * z2

        r = np.concatenate((r_left_flat, r_slope, r_bottom_flat,
                            np.flip(r_bottom_flat) * - 1, np.flip(r_slope) * - 1, np.flip(r_left_flat) * -1))
        z = np.concatenate((z_left_flat, z_slope, z_bottom_flat,
                            np.flip(z_bottom_flat), np.flip(z_slope), z_left_flat))
    else:
        r = np.concatenate((r_left_flat, r_slope, np.flip(r_slope) * - 1, np.flip(r_left_flat) * -1))
        z = np.concatenate((z_left_flat, z_slope, np.flip(z_slope), z_left_flat))

    return r, z

def plot_profile_with_hole(r, z, radius_hole, title=None, equal_axes=False, path_save=None):
    r_hole = r[np.abs(r) > radius_hole]
    z_hole = z[np.abs(r) > radius_hole]

    fig, ax = plt.subplots()
    ax.scatter(r, z, s=1, color='k', label='profile', zorder=3.1)
    if len(r_hole) < len(r):
        ax.plot(r_hole, z_hole, 'mo', ms=1, label='w/o hole', zorder=3.2)
        ax.plot(r_hole, z_hole, 'b-', lw=0.5, label='w/ hole crossing', zorder=3)
    ax.grid(alpha=0.25)
    if equal_axes:
        ax.axis('equal')
    if title is not None:
        ax.set_title(title)
    ax.legend(loc='upper center')
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(path_save, dpi=300, facecolor='w', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    BASE_DIR = ('/Users/mackenzie/Library/CloudStorage/Box-Box/2024/zipper_paper/Testing/Zipper Actuation/'
                '010120205_W101-A1_RepresentativeExample')

    use_pre_computed_values = False
    if use_pre_computed_values:
        r1 = np.arange(-2000, -1500, 1)
        z1 = np.zeros_like(r1)

        r2 = np.arange(-1500, 0, 1)
        z2 = np.linspace(0, -264, 1500)

        #r3 = np.arange(0, 1500, 1)
        #z3 = np.arange(1500, 0, -1)

        #r4 = np.arange(1500, 2000, 1)
        #z4 = np.zeros_like(r4)

        r = np.concatenate((r1, r2, np.flip(r2) * - 1, np.flip(r1) * -1))
        z = np.concatenate((z1, z2, np.flip(z2), z1))

        radius_hole = 731.49 / 2
        r_hole = r[np.abs(r) > radius_hole]
        z_hole = z[np.abs(r) > radius_hole]

        plt.plot(r, z)
        plt.plot(r_hole, z_hole, '--')
        plt.grid(alpha=0.25)
        plt.axis('equal')
        plt.show()


    compute_using = False
    if compute_using:
        # INPUTS
        diameter = 3000  # units: microns
        angle = 54.7  # units: degrees  (KOH etched sidewall = 54.7)
        thickness = 500  # units: microns
        radius_hole = 500  # units: microns
        fid = 50
        save_id_all = 'representative_sweep_angle'

        df_profiles = []
        for fid in [0, 1, 2, 3]:
            angle = np.round(54.7 / (fid + 1), 1)

            # --- CALCULATE GEOMETRY
            """ There are two points that we need to calculate:
            1. Top left corner of slope: (r = -1/2 diameter, z = 0)
            2. Bottom right corner of slope: (r <= 0, z = thickness)
            """
            # We will always know the first point from the inputs
            r1, z1 = -diameter / 2, 0

            # We need to calculate second point from radius, slope, and thickness.
            # horizontal length spanned by slope
            dX = thickness / np.tan(np.deg2rad(angle))
            # horizontal length of hole
            if diameter / 2 - dX > 0:
                d0X = diameter / 2 - dX
            else:
                thickness = np.round(np.abs(r1) * np.tan(np.deg2rad(angle)))
                d0X = 0
                print("Slope extends farther than radius. New thickness: ", thickness, "um.")


            # Second point
            r2, z2 = -d0X, -thickness

            # --- Creat discretized space
            r, z = discretize_profile(r1, z1, r2, z2, r_spacing=1)

            # ensure no points where hole is (this is just for visualization)
            save_id = 'fid{}_Dia{}um_{}degSidewall_{}umThick_RHole{}um'.format(fid, diameter, angle, thickness, radius_hole)
            plot_profile_with_hole(r, z, radius_hole, title=save_id,
                                   equal_axes=False, path_save=join(BASE_DIR, save_id + '.png'))

            df_profile = pd.DataFrame({'r': r, 'z': z})
            df_profile.insert(0, 'fid', fid)
            for c, x in zip(['diameter', 'angle', 'thickness', 'radius_hole'], [diameter, angle, thickness, radius_hole]):
                df_profile[c] = x
            df_profiles.append(df_profile)

        df_profiles = pd.concat(df_profiles)
        df_profiles.to_excel(join(BASE_DIR, save_id_all + '.xlsx'), index=False)

        a = 1