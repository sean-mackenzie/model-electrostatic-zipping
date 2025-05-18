# tests/test_model_sweep.py

# imports
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

from utils import shapes
from archived.old_model_energy_minimization import legacy_solve_energy_iterative_shape_function


# functions: plotting

def complete_erf_profile(px, py, width, dia_flat):
    """ px_revolved, py_revolved = complete_erf_profile(px, py, width, dia_flat) """
    px_revolved = np.flip(px) + width / 2 + dia_flat / 2
    py_revolved = py

    px_revolved = np.append(px_revolved, px[-1])
    py_revolved = np.append(py_revolved, py[-1])

    return px_revolved, py_revolved


def plot_profile_fullwidth(full, use, shape_function, width, depth, num_segments, shape_x0, dia_flat,
                           ax=None, ylbl=True, ls=None):
    """ ax = plot_profile_fullwidth(full, use, shape_function, width, depth, num_segments, shape_x0, dia_flat, ax) """
    # px, py = get_erf_profile(X, Z, num_segments, shape_x0, dia_flat)
    px, py = shape_function(width, depth, num_segments, shape_x0, dia_flat)

    # plot

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    if full is True:
        if use == 'erf':
            px_revolved, py_revolved = complete_erf_profile(px, py, width, dia_flat)
        elif use == 'flat':
            raise ValueError("Not implemented yet.")
            px_revolved, py_revolved = complete_flat_profile(px, py, width, theta=shape_x0)
        else:
            raise ValueError('must be: [erf, flat]')

        if ls is not None:
            ax.plot(px * 1e3, py * 1e6, ls)
            ax.plot(px_revolved * 1e3, py_revolved * 1e6, ls)
        else:
            ax.plot(px * 1e3, py * 1e6, ls='-', color='tab:blue')
            ax.plot(px_revolved * 1e3, py_revolved * 1e6, ls='-', color='tab:blue', label='Cross-section')
    else:
        ax.plot(px * 1e3, py * 1e6, ls='-', color='tab:blue', label='Zipping Path')
        ax.set_title('Dia.={}, Dia. Flat={}'.format(X, dia_flat))

    if ylbl:
        ax.set_ylabel('z (um)')

    ax.grid(alpha=0.25)
    ax.set_xlabel('r (mm)')

    if ls is None:
        ax.legend()

    if ax is None:
        plt.tight_layout()
        plt.show()
        plt.close()
    else:
        return ax


def plot_sweep_z_by_v(df_roots, path_save, save_id):
    mrkrs = ['b-', 'r--', 'g-.', 'k-']

    fig, ax = plt.subplots(figsize=(4.5, 3))

    for j in range(len(df_roots)):
        ax.plot(df_roots[j].U, df_roots[j].z * 1e6, mrkrs[j], label=vs[j])

    ax.set_xlabel('V (V)', fontsize=14)
    ax.set_ylabel('z (um)', fontsize=14)
    ax.legend(title=k)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, save_id + '_sweep_z-by-v.png'), dpi=300, facecolor='w')
    else:
        plt.show()
    plt.close()


def plot_sweep_z_by_v_and_rz_profile():
    mrkrs = ['b-', 'r--', 'g-.', 'k-']

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(8.5, 3))

    for j in range(len(df_roots)):
        vval = vs[j]
        ls = mrkrs[j]

        ax2 = plot_profile_fullwidth(full=True,
                                     use='erf',
                                     shape_function=shape_function,
                                     width=vval,
                                     depth=Z,
                                     num_segments=num_segments,
                                     shape_x0=shape_x0,
                                     dia_flat=dia_flat,
                                     ax=ax2,
                                     ylbl=False,
                                     ls=ls,
                                     )

        ax1.plot(df_roots[j].U, df_roots[j].z * -1e6, ls, label=vs[j])

    ax2.set_xlabel('r (mm)', fontsize=14)
    ax2.set_title("x0={}, flat={}".format(shape_x0, dia_flat * 1e3))

    ax1.set_xlabel('V (V)', fontsize=14)
    ax1.set_ylabel('z (um)', fontsize=14)
    ax1.legend(title=k)
    ax1.grid(alpha=0.25)
    ax1.set_title('Pre-stretch={}'.format(pre_stretch))
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, save_id + '_sweep_z-by-v_and_rz-profile.png'), dpi=300, facecolor='w')
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    # ---

    """
    INDIRECT INPUTS
    """

    # -

    """
    dict_actuator = {
    'shape': shape,
    'shape_function: shape_function,
    'diameter': X,
    'depth': Z,
    'x0': shape_x0,
    'dia_flat': dia_flat,
    'membrane_thickness': t,
    'pre_stretch': pre_stretch,
    'dielectric_thickness': t_diel,
    'profile_x': input_profile_x,
    'profile_z': input_profile_z,
    }
    """
    shape = 'circle'
    shape_function = shapes.get_erf_profile
    X = 1.5e-3  # chamber width = 2 mm
    Z = 200e-6  # chamber depth = 100 microns
    dia_flat = 0.125e-3
    shape_x0 = [1.25, 1.25]
    t = 20e-6  # membrane thickness = 25 microns
    pre_stretch = 1.1  # no pre-stretch
    t_diel = 1e-6
    input_profile_x = np.arange(3)  # pass in profile_x, which can be external
    input_profile_z = np.arange(3)  # pass in profile_z, which can be external

    """
    dict_material = {
    'mu': mu_memb, 
    'Jm': J_memb, 
    'eps_r_memb': eps_r_memb,
    'eps_r_diel': eps_r_diel, 
    'surface_roughness_diel': surface_roughness_diel,
    }
    """
    # material
    Youngs_Modulus = 2e6
    mu_memb = Youngs_Modulus / 3  # 0.42e6
    J_memb = 80.4
    eps_r_memb = 3.0
    eps_r_diel = 3.0
    surface_roughness_diel = 0.0

    # ---

    """
    DIRECT INPUTS
    """

    # -

    # design of actuator (independent variables)
    config = 'MoB'

    dict_actuator = {
        'shape': shape,
        'shape_function': shape_function,
        'diameter': X,
        'depth': Z,
        'x0': shape_x0,
        'dia_flat': dia_flat,
        'membrane_thickness': t,
        'pre_stretch': pre_stretch,
        'dielectric_thickness': t_diel,
        'profile_x': input_profile_x,
        'profile_z': input_profile_z,
    }
    dict_material = {
        'mu': mu_memb,
        'Jm': J_memb,
        'eps_r_memb': eps_r_memb,
        'eps_r_diel': eps_r_diel,
        'surface_roughness_diel': surface_roughness_diel,
    }

    # identifier (independent variable? Or, can be made dependent?)
    save_id = 'SeekStrain_tmemb200um__sweep-Diameter'

    # number of steps taken by solver (can be made a dependent variable or hard-coded?)
    num_segments = 3500

    # voltage (can be made a dependent variable or hard-coded?)
    U_is = np.arange(5, 400, 10)

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

    path_save = '/Users/mackenzie/Desktop/zipper_paper/Modeling/apply model to my wafers/test_model-electrostatic-zipping'

    # ------------------------------------------------------------------------------------------------------------------
    # RUN FUNCTION - SOLVE AND PLOT Z-BY-V
    # ------------------------------------------------------------------------------------------------------------------

    # setup
    d = dict_actuator  # dict_actuator dict_material
    k = 'diameter'  # 'pre_stretch' 'dia_flat' 'x0'
    vs = [1.5e-3, 2e-3, 2.5e-3]

    # solve
    df_roots = []
    for i in range(len(vs)):
        d.update({k: vs[i]})
        if k in ['shape_function', 'diameter', 'depth', 'x0', 'dia_flat']:
            profile_x, profile_z = shapes.get_erf_profile_from_dict(dict_actuator=d, num_points=num_segments)
            d.update({'profile_x': profile_x, 'profile_z': profile_z})

        """
        plt.plot(dict_actuator['profile_x'], dict_actuator['profile_z'])
        plt.show()
        plt.close()
        raise ValueError()
        """

        df_roots1 = legacy_solve_energy_iterative_shape_function(config,
                                                          dict_actuator,
                                                          dict_material,
                                                          num_segments,
                                                          U_is,
                                                          append_dfs=append_dfs,
                                                          export_excel=export_excel,  # 100,
                                                          save_id=save_id)
        df_roots.append(df_roots1)

    # plot
    plot_sweep_z_by_v(df_roots, path_save, save_id)

    # the function below would need to be updated to include all inputs (there are many)
    # plot_sweep_z_by_v_and_rz_profile()

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
    for i in range(len(vs)):
        d.update({k: vs[i]})

        if k in ['shape_function', 'diameter', 'depth', 'x0', 'dia_flat']:
            profile_x, profile_z = shapes.get_erf_profile_from_dict(dict_actuator=d, num_points=num_segments)
            d.update({'profile_x': profile_x, 'profile_z': profile_z})

        dfs1, df_roots1 = legacy_solve_energy_iterative_shape_function(config,
                                                                dict_actuator,
                                                                dict_material,
                                                                num_segments,
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

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(6, 5))

    for j in range(len(dfs)):

        df = dfs[j][0]
        vval = vs[j]
        ls = mrkrs[j]

        df['strain_xy'] = df['stretch_i'] / df['stretch_i'].iloc[0]
        df['strain_z'] = df['t_i'] / df['t_i'].iloc[0]
        df['strain_z_inv'] = 1 / df['strain_z']

        if export_excel_strain:
            df.to_excel(join(path_save, save_id + '_strain-by-z_{}_{}.xlsx'.format(k, vval)))

        # plot

        ax1.plot(df['dZ'] * 1e6, df['t_i'] * 1e6, ls, label=vval)

        if ls == 'b-':
            ax2.plot(df['dZ'] * 1e6, df['strain_xy'], color=ls[0], ls='-', label='in-plane')
            ax2.plot(df['dZ'] * 1e6, df['strain_z_inv'], color=ls[0], ls='--', label='out-of-plane')
        else:
            ax2.plot(df['dZ'] * 1e6, df['strain_xy'], color=ls[0], ls='-')
            ax2.plot(df['dZ'] * 1e6, df['strain_z_inv'], color=ls[0], ls='--')

        # -

    ax1.set_ylabel('Memb. Thickness', fontsize=14)
    ax1.grid(alpha=0.25)
    ax1.legend(title=k)

    ax2.set_ylabel('Strain', fontsize=14)
    ax2.set_xlabel('z (um)', fontsize=14)
    ax2.grid(alpha=0.25)
    ax2.legend()

    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, save_id + '_strain-by-z.png'), dpi=300, facecolor='w')
    else:
        plt.show()
    plt.close()