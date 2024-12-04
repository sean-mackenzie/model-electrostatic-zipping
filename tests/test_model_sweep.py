# tests/test_model_sweep.py

# imports
from os.path import join
import numpy as np
import pandas as pd
from scipy.special import erf
import matplotlib.pyplot as plt

# constants
eps0 = 8.854e-12

# functions: in-plane geometry
def surface_area(l, shape):
    if shape == 'square':
        SA = l * l
    elif shape == 'circle':
        SA = np.pi * l ** 2 / 4
    return SA

def perimeter(l, shape):
    if shape == 'square':
        P = 4 * l
    elif shape == 'circle':
        P = np.pi * l
    return P

# functions: cross-section geometry
def get_erf_profile(diameter, depth, num_points, x0, diameter_flat):
    """ diameter=4e-3, depth=65e-6, num_points=200, x0=1.5, diameter_flat=1e-3 """
    if isinstance(x0, (list, np.ndarray)):
        x1, x2 = x0[0], x0[1]
    else:
        x1, x2 = x0, x0

    erf_x = np.linspace(-x1, x2, num_points)
    erf_y = erf(erf_x)

    norm_erf_x = (erf_x - erf_x[0]) / (x1 + x2)
    norm_erf_y = (erf_y - erf_y[0])
    norm_erf_y = norm_erf_y / -norm_erf_y[-1]

    profile_x = norm_erf_x * diameter / 2
    profile_z = norm_erf_y * depth

    profile_x = profile_x * (diameter - diameter_flat) / diameter

    return profile_x, profile_z


def get_erf_profile_from_dict(dict_actuator, num_points):
    """ diameter=4e-3, depth=65e-6, num_points=200, x0=1.5, diameter_flat=1e-3 """
    diameter = dict_actuator['diameter']
    depth = dict_actuator['depth']
    x0 = dict_actuator['x0']
    diameter_flat = dict_actuator['dia_flat']

    if isinstance(x0, (list, np.ndarray)):
        x1, x2 = x0[0], x0[1]
    else:
        x1, x2 = x0, x0

    erf_x = np.linspace(-x1, x2, num_points)
    erf_y = erf(erf_x)

    norm_erf_x = (erf_x - erf_x[0]) / (x1 + x2)
    norm_erf_y = (erf_y - erf_y[0])
    norm_erf_y = norm_erf_y / -norm_erf_y[-1]

    profile_x = norm_erf_x * diameter / 2
    profile_z = norm_erf_y * depth

    profile_x = profile_x * (diameter - diameter_flat) / diameter

    return profile_x, profile_z

# functions: energy
def mechanical_energy_density_Gent(mu, J, l):
    return -1 * (mu * J / 2) * np.log(1 - ((2 * l ** 2 + 1 / l ** 4 - 3) / J))

def capacitance_surface_roughness(eps_r, A, d, R0):
    return eps0 * A / R0 * np.log(eps_r * R0 / d + 1)

def electrostatic_energy_density_SR(eps_r, A, d, U, R0):
    if R0 == 0:
        R0 = 1e-9
    return -0.5 * U ** 2 * capacitance_surface_roughness(eps_r, A, d, R0)

# function: model solver
def solve_energy_iterative_shape_function(config, dict_actuator, dict_material,
                                          num_segments, U_is, append_dfs=False, export_excel=False,
                                          save_id='arb', silence=True):
    """
    Notes
    """

    actuator_shape = dict_actuator['shape']
    X = dict_actuator['diameter']
    Z = dict_actuator['depth']
    t = dict_actuator['membrane_thickness']
    pre_stretch = dict_actuator['pre_stretch']

    t_diel = dict_actuator['dielectric_thickness']
    eps_r_diel = dict_material['eps_r_diel']
    surface_roughness = dict_material['surface_roughness_diel']

    mu = dict_material['mu']
    J = dict_material['Jm']
    eps_r_memb = dict_material['eps_r_memb']

    # ----------- Set up shape function stuff ---------------------------------

    """ Stuff that shouldn't be needed any longer """
    # theta = dict_actuator['sidewall_angle']
    # Ls = Z / np.cos(np.deg2rad(theta))  # Ls: the length (hypotenuse) of the sidewall
    # LsX = Z * np.tan(np.deg2rad(theta))  # LsX: the x-direction length of the sidewall
    # Ltot = X - 2 * LsX + 2 * Ls  # Ltot: the total length of the chamber profile (sidewall + flat)
    # 1. dL: the length (hypotenuse) of each segment
    # dL = Ls / num_segments

    """ Stuff that needs to be calculated differently now """
    if 'profile_x' in dict_actuator.keys():
        profile_x = dict_actuator['profile_x']
        profile_z = dict_actuator['profile_z']
    else:
        actuator_shape_function = dict_actuator['shape_function']
        profile_x, profile_z = actuator_shape_function(X,
                                                       Z,
                                                       num_segments,
                                                       dict_actuator['x0'],
                                                       dict_actuator['dia_flat'])

    """ Stuff that needs to be calculated every step now """
    # 2. dX: differential x-direction length
    # dX = dL * np.sin(np.deg2rad(theta))
    # 3. dZ: differential z-direction length
    # dZ = dL * np.cos(np.deg2rad(theta))

    # -----------------------------------------------------------------------------

    # -

    # --- SOLVER

    root_init_deflection = 0

    U_firsts = []
    root_firsts = []
    dfs = []

    for U_i in U_is:

        dL_i = 0  # dL_i: current total length (hypotenuse) of zipped segments
        dX_i = 0  # dX_i: current x-direction length of zipped segments
        dZ_i = 0  # dZ_i (z_i): current z-direcetion length of zipped segments

        x_i = X  # (X_i) moving width of flat membrane
        t_i = t  # moving membrane thickness
        L_i = x_i * pre_stretch  # (was x_i) moving length of membrane

        stretch_i = L_i / x_i  # moving stretch in flat membrane
        Vol_i = surface_area(l=x_i, shape=actuator_shape) * t_i  # moving volume of flat membrane

        SA_sum_i = 0  # moving summation of zipped surface area

        Em_seg_sum_i = 0  # moving summation of mechanical energy over all zipped segments
        Es_seg_sum_i = 0  # moving summation of electrostatic energy over all zipped segments

        res = []
        for i in np.arange(1, len(profile_x)):

            """ Stuff that needs to be calculated every step now """
            # 1. evaluate moving positions

            dX = profile_x[i] - profile_x[i - 1]
            dZ = (profile_z[i] - profile_z[i - 1]) * -1
            dL = np.sqrt(dX ** 2 + dZ ** 2)

            dX_i += dX  # dX_i: current x-direction length of zipped segments
            dZ_i += dZ  # dZ_i (z_i): current z-direcetion length of zipped segments
            dL_i += dL  # dL_i: current total length (hypotenuse) of zipped segments

            # ---

            # 2. evaluate current zipped segment
            perimeter_i = perimeter(l=x_i, shape=actuator_shape)
            surface_area_i = dL * perimeter_i
            vol_i = surface_area_i * t_i

            SA_sum_i += surface_area_i

            # energy stored in zipped segment

            # mechanical
            Em_seg_i = vol_i * mechanical_energy_density_Gent(mu=mu, J=J, l=stretch_i)

            # electrostatic
            if config == 'MoT':
                # Es_seg_i = electrostatic_energy_density(eps_r=eps_r_memb, A=surface_area_i, d=t_i, U=U_i)
                Es_seg_i = electrostatic_energy_density_SR(eps_r=eps_r_memb, A=surface_area_i, d=t_i, U=U_i,
                                                           R0=surface_roughness)
            elif config == 'MoB':
                # Es_seg_i = electrostatic_energy_density(eps_r=eps_r_diel, A=surface_area_i, d=t_diel, U=U_i)
                Es_seg_i = electrostatic_energy_density_SR(eps_r=eps_r_diel, A=surface_area_i, d=t_diel, U=U_i,
                                                           R0=surface_roughness)
            else:
                raise ValueError("Not one of [MoT, MoB]")

            # ---

            # data formulating the zipped segment
            res_zip_i = [i,
                         dL_i, dX_i, dZ_i,
                         L_i, x_i, t_i, stretch_i,
                         perimeter_i, surface_area_i, vol_i,
                         Em_seg_i, Es_seg_i,
                         ]

            # ---

            # calculate the original length (t = 50 microns) of this flat section
            L_i_eff0 = np.sqrt(x_i ** 2 * t_i / t)

            # 3. reevaluate flat membrane (calculate new stretch)
            x_i = x_i - 2 * dX  # current width - 2 * differential x-length per segment
            # L_i = x_i * pre_stretch + 2 * dL   OLD METHOD: CHANGED ON 10/9/24
            L_i = (x_i + 2 * dL) * pre_stretch
            stretch_i = L_i / L_i_eff0

            Vol_i = Vol_i - vol_i
            t_i = Vol_i / surface_area(l=x_i, shape=actuator_shape)

            ### mechanical energy of flat membrane
            Em_flat_i = Vol_i * mechanical_energy_density_Gent(mu=mu, J=J, l=stretch_i)

            res_flat_i = [L_i_eff0, L_i, x_i, t_i, stretch_i, Vol_i, Em_flat_i, ]

            # ---

            # total energy stored in flat + zipped segments

            # Total mechanical
            Em_seg_sum_i += Em_seg_i  # total mechanical in zipped segments
            Em_tot_i = Em_seg_sum_i + Em_flat_i

            # Total Electrical
            Es_seg_sum_i += Es_seg_i  # total electrical in zipped segments

            # Total energy
            E_tot_i = Em_tot_i + Es_seg_sum_i

            # ---
            res_e_i = [Em_seg_sum_i, Em_tot_i, Es_seg_sum_i, E_tot_i]

            res_i = res_zip_i + res_flat_i + res_e_i
            res.append(res_i)

        # dataframe
        columns_zip = ['step',
                       'dL', 'dX', 'dZ',
                       'L_i', 'x_i', 't_i', 'stretch_i',
                       'perimeter_i', 'sa_i', 'vol_i',
                       'Em_seg_i', 'Es_seg_i',
                       ]
        columns_flat = ['L0_f', 'L_f', 'x_f', 't_f', 'stretch_f', 'vol_f', 'Em_flat_i']
        columns_energy = ['Em_seg_sum_i', 'Em_tot_i', 'Es_seg_sum_i', 'E_tot_i']
        columns = columns_zip + columns_flat + columns_energy

        df = pd.DataFrame(np.array(res),
                          columns=columns)

        if export_excel == True:
            print("Exporting")
            df.to_excel('{}_{}_{}_U={}V.xlsx'.format(save_id, config, actuator_shape, U_i))
        elif isinstance(export_excel, (int, float)):
            if U_i == export_excel:
                df.to_excel('{}_{}_{}_U={}V.xlsx'.format(save_id, config, actuator_shape, U_i))

        # ---

        # --- find first minimina of total energy
        pxn = df.dZ.to_numpy()[1:-1]
        py3n = df.E_tot_i.diff().to_numpy()[1:-1]
        pf12 = np.poly1d(np.polyfit(pxn, py3n, 12))
        # pfy = pf12(pxn)

        # fig, ax = plt.subplots()
        # ax.plot(pxn, py3n, 'k-')
        # ax.plot(pxn, pf12(pxn), 'r--')
        # plt.show()
        # plt.close()

        try:
            roots = np.roots(pf12)
            roots = roots[np.imag(roots) == 0]
            roots = np.real(roots)
            roots = roots[(roots > df.dZ.min()) & (roots < df.dZ.max())]
            roots.sort()
            root_first = roots[0]

            U_firsts.append(U_i)
            root_firsts.append(root_first)

            root_init_deflection = 1

        except IndexError:
            if root_init_deflection == 0:
                nan_root = 0
            elif root_init_deflection == 1:
                nan_root = Z

            U_firsts.append(U_i)
            root_firsts.append(nan_root)

        # ---

        if append_dfs:
            dfs.append(df)
        else:
            del df

    # package into dataframe and export
    df_roots = pd.DataFrame(np.vstack([U_firsts, root_firsts]).T, columns=['U', 'z'])
    # df.to_excel('z_by_U__t={}um.xlsx'.format(int(np.round(t*1e6))))

    if append_dfs:
        return dfs, df_roots
    else:
        return df_roots


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
    shape_function = get_erf_profile
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
        profile_x, profile_z = get_erf_profile_from_dict(dict_actuator=d,
                                                         num_points=num_segments)
        d.update({'profile_x': profile_x, 'profile_z': profile_z})

        df_roots1 = solve_energy_iterative_shape_function(config,
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
        profile_x, profile_z = get_erf_profile_from_dict(dict_actuator=d,
                                                         num_points=num_segments)
        d.update({'profile_x': profile_x, 'profile_z': profile_z})

        dfs1, df_roots1 = solve_energy_iterative_shape_function(config,
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