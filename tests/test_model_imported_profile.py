# tests/test_model_sweep.py

# imports
import os
from os.path import join
import numpy as np
import pandas as pd
from scipy.special import erf
import matplotlib.pyplot as plt

from utils import energy, plotting, shapes
from utils.empirical import dict_from_tck
from model_energy_minimization import solve_energy_iterative_shape_function


def plot_sweep_z_by_v(df_roots, key, vals, path_save, save_id, key_title=None):
    mrkrs = ['b-', 'r--', 'g-.', 'k-']

    fig, ax = plt.subplots(figsize=(4.5, 3))

    for j in range(len(df_roots)):
        ax.plot(df_roots[j].U, df_roots[j].z * 1e6, mrkrs[j], label=vals[j])

    ax.set_xlabel('V (V)', fontsize=14)
    ax.set_ylabel('z (um)', fontsize=14)
    if key_title is not None:
        ax.legend(title=key_title)
    else:
        ax.legend(title=key)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, save_id + f'_sweep-{key}_z-by-v.png'), dpi=300, facecolor='w')
    else:
        plt.show()
    plt.close()



if __name__ == '__main__':
    # general inputs
    base_dir = '/Users/mackenzie/Desktop/zipper_paper'

    # specific inputs
    wid = 13  # 14
    fid = 6  # 3
    depth = 200  # 195
    radius = 1767  # 1772
    max_dz_for_strain_plot = depth - 3
    units = 1e-6
    num_segments = 3500  # NOTE: this isn't necessarily the final number of solver segments

    save_dir_ = join(base_dir, 'Modeling/apply model to my wafers/first-pass_by-wid')
    save_dir = join(save_dir_, f'wid{wid}')
    save_id = 'wid{}_fid{}'.format(wid, fid)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # read_dir = join(base_dir, 'Fabrication/grayscale')
    # filepath_tck = 'w{}/results/profiles_tck/fid{}_tc_k=3.xlsx'
    # fp_tck = join(read_dir, filepath_tck.format(wid, fid))
    dict_fid = dict_from_tck(wid, fid, depth, radius, units, num_segments, fp_tck=None)

    # ---
    config = 'MoB'
    shape = 'circle'
    diameter = dict_fid['radius'] * 2 * units
    depth = depth * units
    t = 20e-6  # membrane thickness = 25 microns
    pre_stretch = 1.025  # pre-stretch
    t_diel = 2e-6

    # material
    Youngs_Modulus = 2e6
    # mu_memb = Youngs_Modulus / 3  # 0.42e6
    J_memb = 80.4
    eps_r_memb = 3.0
    eps_r_diel = 3.0
    surface_roughness_diel = 0.0

    # voltage (can be made a dependent variable or hard-coded?)
    U_is = np.arange(5, 255, 5)

    # profile to use
    use_tck = True
    if use_tck:
        px, py = dict_fid['r'], dict_fid['z']
        #py = py[px > 10e-6]
        #px = px[px > 10e-6]

    else:
        diameter = 1.55e-3
        depth = 200e-6
        px, py = shapes.get_erf_profile(diameter=diameter,#dict_fid['radius'] * 2 * units,
                                        depth=depth,#dict_fid['depth'] * units,
                                        num_points=num_segments,
                                        x0=[1.25, 1.25],
                                        diameter_flat=0.125e-3)
    #"""
    plt.plot(px * 1e3, py * 1e6)
    plt.xlabel('mm')
    plt.ylabel('um')
    plt.grid(alpha=0.25)
    plt.suptitle(save_id)
    plt.savefig(join(save_dir, save_id + '.png'), facecolor='w')
    plt.show()
    plt.close()
    #raise ValueError()
    #"""


    # dict_actuator: shape, profile_x, profile_z, membrane_thickness, pre_stretch, dielectric_thickness
    dict_actuator = {
        'shape': shape,
        'diameter': diameter,
        'depth': depth,
        'membrane_thickness': t,
        'pre_stretch': pre_stretch,
        'dielectric_thickness': t_diel,
        'profile_x': px,
        'profile_z': py,
    }
    dict_material = {
        'E': Youngs_Modulus,  # 'mu': mu_memb,
        'Jm': J_memb,
        'eps_r_memb': eps_r_memb,
        'eps_r_diel': eps_r_diel,
        'surface_roughness_diel': surface_roughness_diel,
    }

    # execution modifiers

    append_dfs = False
    """
    If False,
        INPUTS: U_is is an array of values
        RETURN: df_roots1
        POST-PROCESSING: use df_roots1 to plot z vs. V

    If True, 
        INPUTS: U_is is a single value (presumably, voltage where 100% pull-in occurs)
        RETURN: dfs, df_roots
        POST-PROCESSING: use dfs[0] to calculate strain, then plot
    """

    export_excel = False  # Note: if U_is is array, export_excel can be a U_i value (e.g., 100) to export only that
    """
    If number,
        export only dataframe corresponding to that voltage value to .xlsx (useful for validation)
    If True,
        export all dataframes to .xlsx (usually, undesirable)
    If False:
        do nothing
    """

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # RUN FUNCTION - SOLVE AND PLOT Z-BY-V
    # ------------------------------------------------------------------------------------------------------------------

    # setup
    d = dict_material  # dict_actuator dict_material
    k = 'E'  # 'pre_stretch' 'dia_flat' 'x0'
    vs = [2e6, 3e6, 4e6]
    k_fig = "E (MPa)"
    vs_fig = [2, 3, 4]
    k_xlsx = 'E'
    vs_xlsx = ['2MPa', '3MPa', '4MPa']

    # solve
    df_roots = []
    for i in range(len(vs)):
        d.update({k: vs[i]})
        df_roots1 = solve_energy_iterative_shape_function(config,
                                                          dict_actuator,
                                                          dict_material,
                                                          U_is,
                                                          append_dfs=append_dfs,
                                                          export_excel=export_excel,  # 100,
                                                          save_id=save_id)

        df_roots.append(df_roots1)

    df_s = []
    for df_, v in zip(df_roots, vs):
        df_[k] = v
        df_s.append(df_)
    df_s = pd.concat(df_s)
    df_s.to_excel(join(save_dir, save_id + '_sweep-{}_z-by-v.xlsx'.format(k)), index=False)

    plot_sweep_z_by_v(df_roots, key=k, vals=vs_fig, path_save=save_dir, save_id=save_id, key_title=k_fig)

    raise ValueError()

    # ------------------------------------------------------------------------------------------------------------------
    # RUN FUNCTION - SOLVE AND PLOT STRAIN-BY-Z
    # ------------------------------------------------------------------------------------------------------------------

    # setup

    # voltage
    U_is = [df_roots[-1].U.iloc[-1]]

    # solve
    append_dfs = True
    export_excel = False  # Note: if U_is is array, export_excel can be a U_i value (e.g., 100) to export only that
    export_excel_strain = True

    dfs = []
    df_roots = []
    for i in [0]: # range(len(vs)):
        d.update({k: vs[i]})
        dfs1, df_roots1 = solve_energy_iterative_shape_function(config,
                                                          dict_actuator,
                                                          dict_material,
                                                          U_is,
                                                          append_dfs=append_dfs,
                                                          export_excel=export_excel,  # 100,
                                                          save_id=save_id)
        dfs.append(dfs1)
        df_roots.append(df_roots1)

    # ---

    # the below plotting method should be packaged into a function
    mrkrs = ['b-', 'r--', 'g-.', 'k-']

    # ---

    # plot
    path_save = save_dir

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(6, 6))

    for j in range(len(dfs)):

        df = dfs[j][0]
        vval = vs[j]
        ls = mrkrs[j]

        df['strain_xy'] = df['stretch_i'] / df['stretch_i'].iloc[0]
        df['strain_z'] = df['t_i'] / df['t_i'].iloc[0]
        df['strain_z_inv'] = 1 / df['strain_z']
        df['disp_r_microns'] = (df['x_i'] * (df['strain_xy'] - 1)) / 2 * 1e6  # divide by 2 = radial displacement

        if export_excel_strain:
            df.to_excel(join(path_save, save_id + '_strain-by-z_{}_{}.xlsx'.format(k_xlsx, vs_xlsx[j])))

        # plot
        if max_dz_for_strain_plot is not None:
            df = df[df['dZ'] * 1e6 < max_dz_for_strain_plot]

        ax1.plot(df['dZ'] * 1e6, df['t_i'] * 1e6, ls, label=vs_fig[j])

        if ls == 'b-':
            ax2.plot(df['dZ'] * 1e6, df['strain_xy'], color=ls[0], ls='-', label='in-plane')
            ax2.plot(df['dZ'] * 1e6, df['strain_z_inv'], color=ls[0], ls='--', label='out-of-plane')
        else:
            ax2.plot(df['dZ'] * 1e6, df['strain_xy'], color=ls[0], ls='-')
            ax2.plot(df['dZ'] * 1e6, df['strain_z_inv'], color=ls[0], ls='--')

        ax3.plot(df['dZ'] * 1e6, df['disp_r_microns'], ls)

        # -

    ax1.set_ylabel('t (um)', fontsize=14)
    ax1.grid(alpha=0.25)
    ax1.legend(title=k_fig)

    ax2.set_ylabel('Strain', fontsize=14)
    ax2.grid(alpha=0.25)
    ax2.legend()

    ax3.set_ylabel(r'$\Delta r \: (\mu m)$', fontsize=14)
    ax3.set_xlabel('z (um)', fontsize=14)
    ax3.grid(alpha=0.25)

    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, save_id + f'_sweep-{k}_strain-by-z.png'), dpi=300, facecolor='w')
    else:
        plt.show()
    plt.close()