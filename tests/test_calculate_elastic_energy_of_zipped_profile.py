
# imports
import os
from os.path import join
import numpy as np
# from numpy.polynomial import polynomial as P
# from scipy.interpolate import make_splrep, splev, splrep, splder, sproot, BSpline
import pandas as pd
import matplotlib.pyplot as plt

from utils.shapes import surface_area, perimeter
from utils.energy import calculate_stretched_thickness, mechanical_energy_density_Gent, electrostatic_energy_density_SR
from  utils.energy import mechanical_energy_density_metal

from utils import energy, empirical, settings
from utils.empirical import dict_from_tck
from tests.test_manually_fit_tck_to_surface_profile import manually_fit_tck


def calculate_elastic_energy_of_zipped_profile2(config, dict_actuator, dict_material,
                                               U_is, append_dfs=False, export_excel=False,
                                               save_id='arb', path_save=None):
    # Placeholders from old solver
    L_i, L_i_eff0 = 0, 0

    # --- Profile
    actuator_shape = dict_actuator['shape']
    profile_x = dict_actuator['profile_x']
    profile_z = dict_actuator['profile_z']
    X = profile_x.max() * 2
    Z = profile_z.min()

    print("Original X (dict_actuator['diameter']): {}".format(dict_actuator['diameter']))
    print("New X (profile_x.max() * 2): {}".format(X))
    print("Original Z (dict_actuator['depth']): {}".format(dict_actuator['depth']))
    print("New Z (profile_z.max() * 2): {}".format(Z))

    # --- Membrane
    t0 = dict_actuator['membrane_thickness']  # original membrane thickness
    pre_stretch = dict_actuator['pre_stretch']
    J = dict_material['Jm']
    if 'E' in dict_material.keys():
        mu = dict_material['E'] / 3
    elif 'mu' in dict_material.keys():
        mu = dict_material['mu']
    else:
        raise ValueError("Must provide either 'E' (Youngs modulus) or 'mu' (shear modulus).")
    # -

    # --- SOLVER

    root_init_deflection = 0

    U_firsts = []
    root_firsts = []
    dfs = []

    # Electrical Terms
    eps_r_memb = dict_material['eps_r_memb']
    t_diel = dict_actuator['dielectric_thickness']
    eps_r_diel = dict_material['eps_r_diel']
    surface_roughness = dict_material['surface_roughness_diel']
    U_i = 0
    " BEGIN VOLTAGE LOOP HERE "
    # for U_i in U_is:

    dL_i = 0  # dL_i: current total length (hypotenuse) of zipped segments
    dX_i = 0  # dX_i: current x-direction length of zipped segments
    dZ_i = 0  # dZ_i (z_i): current z-direction length of zipped segments

    x_i = X  # (X_i) moving width of flat membrane
    t_i = calculate_stretched_thickness(original_thickness=t0, stretch_factor=pre_stretch)  # moving membrane thickness
    stretch_i = pre_stretch  # moving stretch in flat membrane
    Vol_i = surface_area(l=x_i, shape=actuator_shape) * t_i  # moving volume of flat membrane

    SA_sum_i = 0  # moving summation of zipped surface area
    Em_seg_sum_i = 0  # moving summation of mechanical energy over all zipped segments
    Es_seg_sum_i = 0  # moving summation of electrostatic energy over all zipped segments

    res = []
    for i in np.arange(1, len(profile_x)):
        # --- CALCULATE VALUES FOR ZIPPED SEGMENT OF MEMBRANE
        # 1. evaluate moving positions
        dX = profile_x[i] - profile_x[i - 1]
        dZ = (profile_z[i] - profile_z[i - 1]) * -1
        dL = np.sqrt(dX ** 2 + dZ ** 2)
        dX_i += dX  # dX_i: current x-direction length of zipped segments
        dZ_i += dZ  # dZ_i (z_i): current z-direcetion length of zipped segments
        dL_i += dL  # dL_i: current total length (hypotenuse) of zipped segments
        # ---
        # 2. evaluate current zipped segment
        perimeter_i = perimeter(l=x_i - dX / 2, shape=actuator_shape)
        surface_area_i = dL * perimeter_i
        vol_i = surface_area_i * t_i
        SA_sum_i += surface_area_i
        # 3. --- energy stored in zipped segment
        # mechanical energy stored in differential zipped segment
        Em_seg_i = vol_i * mechanical_energy_density_Gent(mu=mu, J=J, l=stretch_i)
        # electrostatic energy stored in differential zipped segment
        Es_seg_i = handler_electrostatic_energy_density(config,
                                                        eps_r_diel=eps_r_diel, eps_r_memb=eps_r_memb,
                                                        A=surface_area_i,
                                                        t_diel=t_diel, t_memb=t_i,
                                                        U=U_i, R0=surface_roughness)
        # 4. data formulating the zipped segment
        res_zip_i = [i, dL_i, dX_i, dZ_i, L_i, x_i, t_i, stretch_i,
                     perimeter_i, surface_area_i, vol_i,
                     Em_seg_i, Es_seg_i]
        # ---

        # --- CALCULATE THE NEW VALUES FOR SUSPENDED MEMBRANE
        # 1. evaluate moving positions (None)
        # ---
        # 2. evaluate new flat membrane
        x_i = x_i - 2 * dX  # new diameter of suspended membrane
        Vol_i = Vol_i - vol_i  # new volume of suspended membrane
        t_i = Vol_i / surface_area(l=x_i, shape=actuator_shape)  # new thickness of suspended membrane
        stretch_i = np.sqrt(t0 / t_i)  # new stretch of suspended membrane
        # ---
        # 3. energy stored in flat membrane
        # mechanical energy of flat membrane
        Em_flat_i = Vol_i * mechanical_energy_density_Gent(mu=mu, J=J, l=stretch_i)
        # 4. data formulating the flat membrane
        res_flat_i = [L_i_eff0, L_i, x_i, t_i, stretch_i, Vol_i, Em_flat_i, ]
        # ---

        # --- SUM VALUES FOR ZIPPED + FLAT SEGMENTS
        # Total energy stored in flat + zipped segments
        # Total mechanical
        Em_seg_sum_i += Em_seg_i  # total mechanical in zipped segments
        Em_tot_i = Em_seg_sum_i + Em_flat_i
        # Total Electrical
        Es_seg_sum_i += Es_seg_i  # total electrical in zipped segments
        # Total energy
        E_tot_i = Em_tot_i + Es_seg_sum_i
        # Data formulating the total energies of the system
        res_e_i = [Em_seg_sum_i, Em_tot_i, Es_seg_sum_i, E_tot_i]

        # ---
        # Add all results and append to list
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
        df.to_excel(join(path_save, '{}_{}_{}_U={}V.xlsx'.format(save_id, config, actuator_shape, U_i)))
    elif isinstance(export_excel, (int, float)):
        if U_i == export_excel:
            df.to_excel(join(path_save, '{}_{}_{}_U={}V.xlsx'.format(save_id, config, actuator_shape, U_i)))

    # ---

    # --- find first minima of total energy
    pxn = df.dZ.to_numpy()[1:-1]
    py3n = df.E_tot_i.diff().to_numpy()[1:-1]
    pf12 = np.poly1d(np.polyfit(pxn, py3n, 12))
    # pfy = pf12(pxn)

    pxnew = df.dZ.to_numpy()
    pynew = df.E_tot_i.to_numpy()
    # pfnew = np.poly1d(np.polyfit(pxnew, pynew, 12))
    c, stats = P.polyfit(pxnew, pynew, 12, full=True)

    # spl = make_splrep(pxnew, pynew, k=4)
    # spl_derivative = spl.derivative(nu=1)
    do_scipy = False
    if do_scipy:
        save_fig_dir_ = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/zipper_paper/Testing/Zipper Actuation/01102025_W13-D1_C9-0pT/analyses/modeling/25dV'
        s = 0.0
        save_fig_dir = join(save_fig_dir_, 's={}'.format(s))
        if not os.path.exists(save_fig_dir):
            os.makedirs(save_fig_dir)
        spl = splrep(pxnew, pynew, s=s, k=4)
        dspl = splder(spl)
        zeros = sproot(dspl)
        zeros = zeros[np.imag(zeros) == 0]
        # zeros = zeros[(zeros > df.dZ.min()) & (zeros < df.dZ.max())]
        zeros = zeros[(zeros > df.dZ.min() + 1e-6) & (zeros < df.dZ.max() - 2e-6)]
        zeros.sort()
        try:
            root_first = zeros[0]
            root_firsts_scipy.append(root_first)
            raise_error = False
        except IndexError:
            root_first = -1e-6
            raise_error = True

        # tck = splrep(x, y, s=smoothing, k=degree)
        # x2 = np.linspace(x.min(), x.max(), num_points)
        scipy2 = BSpline(*spl)(pxnew)
        dpxnew = pxnew[2:-25]
        dscipy2 = BSpline(*dspl)(dpxnew)

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)

        ax1.plot(df.dZ * 1e6, df.E_tot_i, 'k-', label='data')
        ax1.plot(pxnew * 1e6, scipy2, 'r--', label='scipy fit')
        ax1.legend()

        ax2.plot(pxn * 1e6, py3n, 'k-', label='diff(data)')
        ax2.plot(pxn * 1e6, pf12(pxn), 'g--', label='polyfit(12)')
        ax2.axhline(0, color='gray', linestyle='--', lw=0.5)
        ax2.legend()

        ax3.plot(dpxnew * 1e6, dscipy2, 'r--', label='dscipy: 0={}'.format(np.round(root_first * 1e6, 2)))
        ax3.axhline(0, color='gray', linestyle='--', lw=0.5)
        ax3.legend()

        plt.tight_layout()
        plt.savefig(join(save_fig_dir, '{}V.png'.format(U_i)))
        plt.close()
        if raise_error:
            raise ValueError("No roots found for {} V".format(U_i))
        j = 1
    else:
        pass  # root_firsts_scipy.append(0)

    try:
        roots = np.roots(pf12)
    except np.linalg.LinAlgError:
        # import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(pxn, py3n, 'k-')
        ax.plot(pxn, pf12(pxn), 'r--')
        ax.set_title(U_i)
        plt.show()
        plt.close()
        raise ValueError()

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


def handler_electrostatic_energy_density(config, eps_r_diel, eps_r_memb, A, t_diel, t_memb, U, R0):
    """
    Calculates the electrostatic energy density based on the given configuration
    and physical parameters. The function supports two configurations:
    'MoT' (Membrane or Tissue) and 'MoB' (Membrane or Bulk). Depending on the
    configuration, the appropriate parameters are passed to the
    `electrostatic_energy_density_SR` function for computation.

    Es_seg_i = handler_electrostatic_energy_density(config,
    eps_r_diel=eps_r_diel, eps_r_memb=eps_r_memb, A=surface_area_i, t_diel=t_diel, t_memb=t_i, U=U_i, R0=surface_roughness)

    :param config: Configuration type indicating the calculation mode. It should
        be either 'MoT' or 'MoB'.
    :param eps_r: Relative permittivity of the material.
    :param A: Surface area of the segment (in square meters).
    :param d: Thickness of the layer (in meters).
    :param U: Potential difference across the layer (in volts).
    :param R0: Surface roughness factor.

    :return: Electrostatic energy density for the given segment and configuration
        as calculated by the `electrostatic_energy_density_SR` function.
    :rtype: float

    :raises ValueError: If `config` is neither 'MoT' nor 'MoB'.
    """
    if config == 'MoT':
        # NOT THIS: Es_seg_i = electrostatic_energy_density(eps_r=eps_r_memb, A=surface_area_i, d=t_i, U=U_i)
        Es_seg_i = electrostatic_energy_density_SR(eps_r=eps_r_memb, A=A, d=t_memb, U=U, R0=R0)
        pass
    elif config == 'MoB':
        # NOT THIS: Es_seg_i = electrostatic_energy_density(eps_r=eps_r_diel, A=surface_area_i, d=t_diel, U=U_i)
        Es_seg_i = electrostatic_energy_density_SR(eps_r=eps_r_diel, A=A, d=t_diel, U=U, R0=R0)
        pass
    else:
        raise ValueError("Not one of [MoT, MoB]")
    return Es_seg_i


def calculate_elastic_energy_of_zipped_profile(config, dict_actuator, dict_material, U_is, dict_metal=None):
    a = 1
    """
            I JUST REALIZED!....
            
            Here's the best way to do this:
                1. Run a script that calculates the geometry at every dZ        (length ~ 2000)
                    * (i.e., the volume of the zipped and flat segments for each dZ)
                    ** THIS EFFECTIVELY FORMS A ***FUNCTION*** 
                        The inputs it takes are 
                            (1) functions for energy density of segments
                            (2) the voltage
                    
                    *** So, you only need to export this spreadsheet ONCE! (and it serves for all variations!!!)
                    
                2. Run a script that calculates the energy density of the zipped segments
                    * Effectively apply energy density functions to each volume in ^spreadsheet
                    * You don't need to save anything. You can just "store it in memory" b/c ^spreadsheet is a function
                    * Though, it may be useful to save all the data (energy as a function of U and dZ)
                
                3. Run a handler function that does #2 for (1) bilayer membrane and (2) separate silicone + metal
                    * MOST IMPORTANTLY, we want to plot the (1) and (2) on the same plot
                        *** Hopefully, this will reveal something about how/why/what works and what doesn't
                        or what is "appropriate" and what is not... Ideally, they should be the same, right?
                
    """

    # --- Profile
    actuator_shape = dict_actuator['shape']
    profile_x = dict_actuator['profile_x']
    profile_z = dict_actuator['profile_z']
    X = profile_x.max() * 2
    Z = profile_z.min()

    print("Original X (dict_actuator['diameter']): {}".format(dict_actuator['diameter']))
    print("New X (profile_x.max() * 2): {}".format(X))
    print("Original Z (dict_actuator['depth']): {}".format(dict_actuator['depth']))
    print("New Z (profile_z.max() * 2): {}".format(Z))

    # --- Membrane
    t0 = dict_actuator['membrane_thickness']  # original membrane thickness
    pre_stretch = dict_actuator['pre_stretch']
    J = dict_material['Jm']
    if 'E' in dict_material.keys():
        mu = dict_material['E'] / 3
    elif 'mu' in dict_material.keys():
        mu = dict_material['mu']
    else:
        raise ValueError("Must provide either 'E' (Youngs modulus) or 'mu' (shear modulus).")
    # -
    # Placeholder from old solver
    L_i_eff0 = 0

    # Material properties of metal film
    if dict_metal is None:
        include_metal = False
    else:
        E_metal = dict_metal['E']  # Pa
        nu_metal = dict_metal['poissons_ratio']
        t_metal = dict_metal['thickness']  # meters
        include_metal = dict_metal['include_metal']

    # --- SOLVER

    root_init_deflection = 0

    U_firsts = []
    root_firsts = []
    dfs = []


    # Electrical Terms
    eps_r_memb = dict_material['eps_r_memb']
    t_diel = dict_actuator['dielectric_thickness']
    eps_r_diel = dict_material['eps_r_diel']
    surface_roughness = dict_material['surface_roughness_diel']
    " BEGIN VOLTAGE LOOP HERE "
    U_res = []
    for U_i in U_is:

        dL_i = 0  # dL_i: current total length (hypotenuse) of zipped segments
        dX_i = 0  # dX_i: current x-direction length of zipped segments
        dZ_i = 0  # dZ_i (z_i): current z-direction length of zipped segments

        x_i = X  # (X_i) moving width of surface profile (effectively, diameter)
        t_i = calculate_stretched_thickness(original_thickness=t0, stretch_factor=pre_stretch)  # moving membrane thickness
        stretch_i = pre_stretch  # moving stretch in flat membrane
        Vol_i = surface_area(l=x_i, shape=actuator_shape) * t_i  # moving volume of flat membrane

        L_i  = X  # moving width of flat membrane
        stretch_i_metal = 1.0  # moving pre-stretch of the metal film

        SA_sum_i = 0  # moving summation of zipped surface area
        Em_seg_sum_i = 0  # moving summation of mechanical energy over all zipped segments
        Es_seg_sum_i = 0  # moving summation of electrostatic energy over all zipped segments

        res = []
        for i in np.arange(1, len(profile_x)):
            # --- CALCULATE VALUES FOR ZIPPED SEGMENT OF MEMBRANE
            # 1. evaluate moving positions
            dX = profile_x[i] - profile_x[i - 1]
            dZ = (profile_z[i] - profile_z[i - 1]) * -1
            dL = np.sqrt(dX ** 2 + dZ ** 2)
            dX_i += dX  # dX_i: current x-direction length of zipped segments
            dZ_i += dZ  # dZ_i (z_i): current z-direcetion length of zipped segments
            dL_i += dL  # dL_i: current total length (hypotenuse) of zipped segments
            # ---
            # 2. evaluate current zipped segment
            perimeter_i = perimeter(l=x_i - dX / 2, shape=actuator_shape)
            surface_area_i = dL * perimeter_i
            vol_i = surface_area_i * t_i
            SA_sum_i += surface_area_i
            # 3. --- energy stored in zipped segment
            Em_seg_i_metal = surface_area_i * t_metal * mechanical_energy_density_metal(E=E_metal, nu=nu_metal, l=stretch_i_metal)
            if not include_metal:
                Em_seg_i_metal = 0
            # mechanical energy stored in differential zipped segment
            Em_seg_i = vol_i * mechanical_energy_density_Gent(mu=mu, J=J, l=stretch_i)
            # electrostatic energy stored in differential zipped segment
            Es_seg_i = handler_electrostatic_energy_density(config,
                                                            eps_r_diel=eps_r_diel, eps_r_memb=eps_r_memb,
                                                            A=surface_area_i,
                                                            t_diel=t_diel, t_memb=t_i,
                                                            U=U_i, R0=surface_roughness)

            # 4. data formulating the zipped segment
            res_zip_i = [i, dL_i, dX_i, dZ_i, L_i, x_i, t_i, stretch_i,
                         perimeter_i, surface_area_i, vol_i,
                         Em_seg_i, Em_seg_i_metal, Es_seg_i]
            # ---

            # --- CALCULATE THE NEW VALUES FOR SUSPENDED MEMBRANE
            # 1. evaluate moving positions (None)
            # ---
            # 2. evaluate new flat membrane
            x_i = x_i - 2 * dX                                      # new width of surface profile
            Vol_i = Vol_i - vol_i                                   # new volume of suspended membrane
            t_i = Vol_i / surface_area(l=x_i, shape=actuator_shape) # new thickness of suspended membrane
            stretch_i = np.sqrt(t0 / t_i)                           # new stretch of suspended membrane
            if np.isnan(stretch_i):
                # print("Stretch is NaN. Exiting.")
                pass

            # NEW: METAL FILM
            """ NOTE: this assumes volume of metal does not change. I'm going to test this out first. 
            If necessary, I can modify the volume and energy density function to incorporate volumetric change. 
            (see ChatGPT discussion)
            """
            L_i = L_i - 2 * dL                                      # new width of flat membrane
            stretch_i_metal = x_i / L_i                             # moving pre-stretch of the metal film
            Vol_i_metal = surface_area(l=x_i, shape=actuator_shape) * t_metal  # moving volume of the metal film
            # mechanical energy of the metal film
            Em_flat_i_metal = Vol_i_metal * mechanical_energy_density_metal(E=E_metal, nu=nu_metal, l=stretch_i_metal)
            if not include_metal:
                Em_flat_i_metal = 0
            # ---
            # 3. energy stored in flat membrane
            # mechanical energy of flat membrane
            Em_flat_i = Vol_i * mechanical_energy_density_Gent(mu=mu, J=J, l=stretch_i)
            # 4. data formulating the flat membrane
            res_flat_i = [L_i_eff0, L_i, x_i, t_i, stretch_i, Vol_i, Em_flat_i,
                          stretch_i_metal, Vol_i_metal, Em_flat_i_metal]
            # ---

            # --- SUM VALUES FOR ZIPPED + FLAT SEGMENTS
            # Total energy stored in flat + zipped segments
            # Total mechanical
            Em_seg_sum_i += Em_seg_i + Em_seg_i_metal # total mechanical in zipped segments
            Em_tot_i = Em_seg_sum_i + Em_flat_i + Em_flat_i_metal
            # Total Electrical
            Es_seg_sum_i += Es_seg_i  # total electrical in zipped segments
            # Total energy
            E_tot_i = Em_tot_i + Es_seg_sum_i
            # Data formulating the total energies of the system
            res_e_i = [Em_seg_sum_i, Em_tot_i, Es_seg_sum_i, E_tot_i]

            # ---
            # Add all results and append to list
            res_i = [U_i] + res_zip_i + res_flat_i + res_e_i
            res.append(res_i)
        # stack results for each voltage
        U_res.append(np.vstack(res))

    # dataframe
    columns_zip = ['step',
                   'dL', 'dX', 'dZ',
                   'L_i', 'x_i', 't_i', 'stretch_i',
                   'perimeter_i', 'sa_i', 'vol_i',
                   'Em_seg_i', 'Em_seg_i_metal', 'Es_seg_i',
                   ]
    columns_flat = ['L0_f', 'L_f', 'x_f', 't_f', 'stretch_f', 'vol_f', 'Em_flat_i',
                    'stretch_i_metal', 'Vol_i_metal', 'Em_flat_i_metal']
    columns_energy = ['Em_seg_sum_i', 'Em_tot_i', 'Es_seg_sum_i', 'E_tot_i']
    columns = ['U_i'] + columns_zip + columns_flat + columns_energy
    df = pd.DataFrame(np.vstack(U_res), columns=columns)

    return df


def plot_energy_by_depth(df, path_save, save_id, z_clip=1.5):
    # plot elastic energy vs. depth
    px = 'dZ'
    df = df[df[px] < df[px].max() - z_clip * 1e-6]
    x = df[px] * 1e6

    py1a, py1b = 'Em_seg_i', 'Es_seg_i'
    py2a, py2b = 'Em_seg_sum_i', 'Em_flat_i'
    py3a, py3b = 'Em_tot_i', 'Es_seg_sum_i'
    py4 = 'E_tot_i'

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(6, 8), sharex=True)
    # Energy of segments
    ax1.plot(x, df[py1a], label=py1a)
    ax1.plot(x, df[py1b], label=py1b)
    # (Mechanical) energy of zipped and flat membrane
    ax2.plot(x, df[py2a], label=py2a)
    ax2.plot(x, df[py2b], label=py2b)
    # Total electrostatic and mechanical energy
    ax3.plot(x, df[py3a], label=py3a)
    ax3.plot(x, df[py3b], label=py3b)
    # Total energy
    ax4.plot(x, df[py4], label=py4)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_ylabel('Energy (J)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize='x-small')
    ax4.set_xlabel('Depth (um)')
    plt.tight_layout()
    plt.savefig(join(path_save, save_id), dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()


def calculate_strain_by_z(df):
    df['stretch_ratio'] = df['stretch_f'] / df['stretch_i'].iloc[0]
    df['disp_r_microns'] = (df['stretch_ratio'] - 1) * df['x_f'] / 2 * 1e6  # divide by 2 = radial displacement
    return df


def plot_deformation_by_depth(df, path_save, save_id, dz_less_than_max=2):
    # plot elastic energy vs. depth DZ_LESS_THAN_FOR_STRAIN_PLOT
    px = 'dZ'
    py1 = 't_f'
    py2 = 'stretch_f'
    py3 = 'disp_r_microns'

    df2 = df[df[px] < df[px].max() - dz_less_than_max * 1e-6]
    x12 = df2[px] * 1e6
    y1 = df2[py1] * 1e6
    y2 = df2[py2]

    x3 = df[px] * 1e6
    y3 = df[py3]

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(3.5, 6), sharex=True)

    ax1.plot(x12, y1, label=py1)
    ax1.set_ylabel(r'$t_{memb} \: (\mu m)$')
    ax1.grid(alpha=0.25)

    ax2.plot(x12, y2, label=py2)
    ax2.set_ylabel(r'$\lambda$')
    ax2.grid(alpha=0.25)

    ax3.plot(x3, y3, label=py3)
    ax3.set_ylabel(r'$\Delta r_{o} \: (\mu m)$')
    ax3.grid(alpha=0.25)
    ax3.set_xlabel('Depth (um)')

    plt.tight_layout()
    plt.savefig(join(path_save, save_id), dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()


def find_first_energy_minima(x, y, print_id=''):
    # Compute the discrete difference to approximate the derivative
    dy = np.diff(y)
    # Find where dy changes from negative to positive (indicates a minima)
    minima_index = np.where((dy[:-1] < 0) & (dy[1:] > 0))[0]
    # Get the first minima index (if it exists)
    if len(minima_index) > 0:
        first_minima_x = x[minima_index[0] + 1]  # +1 because np.diff reduces the length by 1
        first_minima_y = y[minima_index[0] + 1]
        print(f"{print_id}: First minima found at "
              f"x = {np.round(first_minima_x * 1e6, 1)}, "
              f"y = {np.round(first_minima_y * 1e6, 4)} uJ")
    else:
        first_minima_x = first_minima_y = np.nan
        print(f"{print_id}: No minima found.")
    return first_minima_x, first_minima_y




if __name__ == '__main__':
    # ---
    """ NOTE: these values are computed via Bulge Test and/or calculate_pre_stretch_from_residual_stress.py."""
    # MEMB_ID = 'C19-30pT_20+10nmAu'
    # MEMB_YOUNGS_MODULUS_BULGE_TEST = 1e6  # Pa
    # MEMB_RESIDUAL_STRESS_BULGE_TEST = 260e3  # Pa
    # EXPERIMENTAL_PRE_STRETCH_NOMINAL = 1.3
    # EXPERIMENTAL_PRE_STRETCH_MEASURED = 1.131
    # GENT_MODEL_COMPUTED_RESIDUAL_STRESS_FROM_PRE_STRETCH_MEASURED = 223e3  # Pa
    # GENT_MODEL_COMPUTED_PRE_STRETCH = 1.156


    # general inputs
    TEST_CONFIG = '03052025_W13-D1_C19-30pT_20+10nmAu'
    WID = 13
    TID = 1
    ROOT_DIR = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/zipper_paper/Testing/Zipper Actuation'
    BASE_DIR = join(ROOT_DIR, TEST_CONFIG)
    ANALYSES_DIR = join(BASE_DIR, 'analyses')
    READ_SETTINGS = join(ANALYSES_DIR, 'settings')
    SAVE_DIR = join(ANALYSES_DIR, 'modeling', 'energy')
    # settings
    FP_SETTINGS = join(READ_SETTINGS, 'dict_settings.xlsx')
    FP_TEST_SETTINGS = join(READ_SETTINGS, 'dict_tid{}_settings.xlsx'.format(TID))
    DICT_SETTINGS = settings.get_settings(fp_settings=FP_SETTINGS, name='settings', update_dependent=False)
    # ---
    FID_PROCESS_PROFILE = DICT_SETTINGS['fid_process_profile']
    FID_OVERRIDE = None
    DICT_SETTINGS_RADIUS = DICT_SETTINGS['radius_microns']
    if FID_OVERRIDE is None:
        FID = DICT_SETTINGS['fid_process_profile']
    else:
        FID = FID_OVERRIDE

    # -------------------------------

    # --- STEP 0. DEFINE "CONSTANTS"
    # Membrane
    MEMB_ORIGINAL_THICKNESS = 20  # (microns)
    MEMB_ORIGINAL_YOUNGS_MODULUS = 1.1e6  # (Pa)    [between 1.0 and 1.2 in literature and from my bulge tests]
    MEMB_ORIGINAL_RELATIVE_PERMITTIVITY = 3.26  #   [according to the literature]
    MEMB_ORIGINAL_GENT_MODEL_J_MEMB = 54  #         [Elastosil 200 um thick: 54 or 16; Maffli et al. NuSil: 80.4]
    # Si+SiO2 Surface Profile
    SURFACE_DIELECTRIC_RELATIVE_PERMITTIVITY = 3.9  # 4/9/25: changed to 3.9; previously 3.0
    SURFACE_DIELECTRIC_SURFACE_ROUGHNESS = 1e-9  # (units: meters) I think it would be fair to vary this

    # --- ---   THE FOLLOWING VALUES SHOULD BE FAITHFUL TO THE DATA (THEY ARE FOR REFERENCE, NOT USED IN MODEL)
    # --- STEP 1. DEFINE THE MEMBRANE
    MEMB_ID = 'C19-30pT_20+10nmAu'
    # --- STEP 2. PRE-STRETCH
    EXPERIMENTAL_PRE_STRETCH_NOMINAL = 1.3
    EXPERIMENTAL_PRE_STRETCH_MEASURED = 1.131
    MEMB_THICKNESS_POST_MEASURED_PRE_STRETCH = np.round(
        energy.calculate_stretched_thickness(MEMB_ORIGINAL_THICKNESS, EXPERIMENTAL_PRE_STRETCH_MEASURED), 2)
    GENT_MODEL_COMPUTED_RESIDUAL_STRESS_FROM_PRE_STRETCH_MEASURED = 223e3  # Pa
    # --- STEP 3. BULGE TEST
    MEMB_YOUNGS_MODULUS_BULGE_TEST = 1.1e6  # Pa
    MEMB_RESIDUAL_STRESS_BULGE_TEST = 260e3  # Pa
    GENT_MODEL_COMPUTED_PRE_STRETCH_FROM_RESIDUAL_STRESS_BULGE_TEST = 1.156
    # ---
    # --- ---   THE FOLLOWING VALUES ARE USED BY THE MODEL (SOME LEEWAY IS ALLOWED, DEPENDING ON A GIVEN SCENARIO)
    # --- STEP A. PHYSICAL PROPERTIES
    # Membrane thickness
    MODEL_USE_PRE_METAL_PRE_STRETCH_FOR_MEMB_THICKNESS = 1.131  # pre-stretch used ONLY to define membrane thickness
    MODEL_USE_MEMB_THICKNESS = np.round(
        energy.calculate_stretched_thickness(MEMB_ORIGINAL_THICKNESS, MODEL_USE_PRE_METAL_PRE_STRETCH_FOR_MEMB_THICKNESS), 2)
    # --- STEP B. MECHANICAL PROPERTIES
    # Effective pre-stretch: accounts for stresses due to (1) pre-stretch, and (2) residual stress from metal dep.
    MODEL_USE_POST_METAL_PRE_STRETCH_FOR_TOTAL_STRESS = 1.1
    MODEL_USE_YOUNGS_MODULUS = 1.2  # (MPa)
    # --- STEP C. ELECTRICAL PROPERTIES
    MODEL_USE_THICKNESS_DIELECTRIC = 2.1  # (microns)
    # Rarely should you change these. If you do, change them here (and not in the constants section)
    MODEL_USE_DIELECTRIC_RELATIVE_PERMITTIVITY = SURFACE_DIELECTRIC_RELATIVE_PERMITTIVITY  # 3.9
    MODEL_USE_SURFACE_ROUGHNESS = SURFACE_DIELECTRIC_SURFACE_ROUGHNESS  # 1e-9
    # --- METAL FILM
    INCLUDE_METAL_ENERGY = True
    E_METAL = 2.71  # units: GPa  (reference: 2.71 GPa for 20nm Au)
    NU_METAL = 0.44
    THICKNESS_METAL = 20  # units: nanometers
    METAL_LAYERS = f'MPTMS+{THICKNESS_METAL}nmAu'
    # ---
    # Solver
    SWEEP_PARAM = 'E'  # 'E', 'pre_stretch', 'Jm'
    SWEEP_VALS = [3.5, 5.5, 7.5, 9.5]  # units for E are MPa (i.e., do not include 1e6 here)
    SWEEP_VOLTAGES = np.arange(25, 270, 5)
    NUM_SEGMENTS = 3000  # NOTE: this isn't necessarily the final number of solver segments
    # Surface profile
    SURFACE_PROFILE_SUBSET = 'right_half'  # 'left_half', 'right_half', 'full'
    TCK_SMOOTHING = 250.0
    MODEL_USE_RADIUS = DICT_SETTINGS_RADIUS + 35

    # -

    # ---   YOU REALLY SHOULDN'T NEED TO CHANGE ANYTHING BELOW
    # -
    print("Membrane thickness after {} pre-stretch: {} microns".format(EXPERIMENTAL_PRE_STRETCH_MEASURED, MEMB_THICKNESS_POST_MEASURED_PRE_STRETCH))
    print("(USED IN MODEL) Membrane thickness after {} pre-stretch: {} microns".format(MODEL_USE_PRE_METAL_PRE_STRETCH_FOR_MEMB_THICKNESS, MODEL_USE_MEMB_THICKNESS))
    # -
    # raise ValueError()
    # -

    # ---

    # -
    SAVE_ID = 'wid{}_fid{}'.format(WID, FID)
    if INCLUDE_METAL_ENERGY:
        SAVE_DIR = join(SAVE_DIR, 'fid{}-{}-s={}_t{}um_E{}_PS{}_Em{}_sweep-{}'.format(
            FID, SURFACE_PROFILE_SUBSET, TCK_SMOOTHING, MODEL_USE_MEMB_THICKNESS, MODEL_USE_YOUNGS_MODULUS,
            MODEL_USE_POST_METAL_PRE_STRETCH_FOR_TOTAL_STRESS, E_METAL, SWEEP_PARAM))
    else:
        SAVE_DIR = join(SAVE_DIR, 'fid{}-{}-s={}_t{}um_E{}_PS{}_sweep-{}'.format(
            FID, SURFACE_PROFILE_SUBSET, TCK_SMOOTHING, MODEL_USE_MEMB_THICKNESS, MODEL_USE_YOUNGS_MODULUS,
            MODEL_USE_POST_METAL_PRE_STRETCH_FOR_TOTAL_STRESS, SWEEP_PARAM))
    # -
    if SWEEP_PARAM == 'E':
        SWEEP_K = 'E'
        SWEEP_VS = np.array(SWEEP_VALS) * 1e6
        SWEEP_K_FIG = "E (MPa)"
        SWEEP_VS_FIGS = np.round(SWEEP_VS * 1e-6, 1)
        SWEEP_K_XLSX = 'E'
        SWEEP_VS_XLSX = [f'{x}MPa' for x in SWEEP_VS_FIGS]
    elif SWEEP_PARAM == 'pre_stretch':
        SWEEP_K = 'pre_stretch'
        SWEEP_VS = SWEEP_VALS
        SWEEP_K_FIG = "Pre-stretch"
        SWEEP_VS_FIGS = SWEEP_VS
        SWEEP_K_XLSX = 'pre_stretch'
        SWEEP_VS_XLSX = SWEEP_VS
    elif SWEEP_PARAM == 'Jm':
        SWEEP_K = 'Jm'
        SWEEP_VS = [16, 54, 80]
        SWEEP_K_FIG = "Jm"
        SWEEP_VS_FIGS = SWEEP_VS
        SWEEP_K_XLSX = 'Jm'
        SWEEP_VS_XLSX = SWEEP_VS
    elif SWEEP_PARAM == 'FINAL':
        SWEEP_K = 'E'
        SWEEP_VS = [MODEL_USE_YOUNGS_MODULUS]
        SWEEP_K_FIG = "E (MPa)"
        SWEEP_VS_FIGS = [np.round(MODEL_USE_YOUNGS_MODULUS / 1e6, 2)]
        SWEEP_K_XLSX = 'E'
        SWEEP_VS_XLSX = ['{} MPa'.format(np.round(MODEL_USE_YOUNGS_MODULUS / 1e6, 2))]
    else:
        raise ValueError('Invalid sweep parameter: {}'.format(SWEEP_PARAM))
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    #  directories
    READ_TCK = join(SAVE_DIR, 'tck')
    for pth in [SAVE_DIR, READ_TCK]:
        if not os.path.exists(pth):
            os.makedirs(pth)
    FN_TCK = 'fid{}_tc_k=3.xlsx'.format(FID)
    FP_TCK = join(READ_TCK, FN_TCK)
    # surface profile
    INCLUDE_THROUGH_HOLE = True
    DF_SURFACE = empirical.read_surface_profile(
        DICT_SETTINGS,
        subset='full',  # this should always be 'full', because profile will get slice during tck
        hole=INCLUDE_THROUGH_HOLE,
        fid_override=FID_OVERRIDE,
    )
    # specific inputs
    DEPTH = DF_SURFACE['z'].abs().max()
    RADIUS = MODEL_USE_RADIUS
    DZ_LESS_THAN_FOR_STRAIN_PLOT = 2
    MAX_DZ_FOR_STRAIN_PLOT = DEPTH - DZ_LESS_THAN_FOR_STRAIN_PLOT
    UNITS = 1e-6
    # ---
    # --- --- MANUALLY FIT TCK AND EXPORT
    SMOOTHING = TCK_SMOOTHING  # 50
    NUM_POINTS = 500
    DEGREE = 3
    # fit tck
    tck, rmin, rmax = manually_fit_tck(df=DF_SURFACE, subset=SURFACE_PROFILE_SUBSET, radius=RADIUS,
                           smoothing=SMOOTHING, num_points=NUM_POINTS, degree=DEGREE,
                           path_save=READ_TCK, show_plots=False)
    DICT_TCK_SETTINGS = {
        'wid': WID,
        'fid': FID,
        'depth': DEPTH,
        'radius': RADIUS,
        'radius_min': rmin,
        'radius_max': rmax,
        'subset': SURFACE_PROFILE_SUBSET,
        'smoothing': SMOOTHING,
        'num_points': NUM_POINTS,
        'degree': DEGREE,
    }
    # export tck
    DF_TCK = pd.DataFrame(np.vstack([tck[0], tck[1]]).T, columns=['t', 'c'])
    DF_TCK_SETTINGS = pd.DataFrame.from_dict(data=DICT_TCK_SETTINGS, orient='index', columns=['v'])
    with pd.ExcelWriter(FP_TCK) as writer:
        for sheet_name, df, idx, idx_lbl in zip(['tck', 'settings'], [DF_TCK, DF_TCK_SETTINGS], [False, True], [None, 'k']):
            df.to_excel(writer, sheet_name=sheet_name, index=idx, index_label=idx_lbl)
    dict_fid = dict_from_tck(WID, FID, DEPTH, RADIUS, UNITS, NUM_SEGMENTS, fp_tck=FP_TCK, r_min=rmin)
    # ---
    # ---
    # surface
    config = 'MoB'
    shape = 'circle'
    diameter = dict_fid['radius'] * 2 * UNITS
    depth = DEPTH * UNITS
    t_diel = MODEL_USE_THICKNESS_DIELECTRIC * UNITS
    eps_r_diel = MODEL_USE_DIELECTRIC_RELATIVE_PERMITTIVITY
    surface_roughness_diel = MODEL_USE_SURFACE_ROUGHNESS  # units: meters
    # membrane
    t = MODEL_USE_MEMB_THICKNESS * UNITS
    pre_stretch = MODEL_USE_POST_METAL_PRE_STRETCH_FOR_TOTAL_STRESS
    Youngs_Modulus = MODEL_USE_YOUNGS_MODULUS * 1e6  # Shear modulus, mu_memb = Youngs_Modulus / 3
    J_memb = MEMB_ORIGINAL_GENT_MODEL_J_MEMB
    eps_r_memb = MEMB_ORIGINAL_RELATIVE_PERMITTIVITY
    # metal
    youngs_modulus_metal = E_METAL * 1e9  # convert to Pa
    poissons_ratio_metal = NU_METAL
    t_metal = THICKNESS_METAL * 1e-9  # convert to m
    include_metal_energy = INCLUDE_METAL_ENERGY
    # test: voltage
    U_is = SWEEP_VOLTAGES

    DICT_MODEL_SETTINGS = {
        'save_id': SAVE_ID,
        'test_config': TEST_CONFIG,
        'wid': WID,
        'fid_tested': FID_PROCESS_PROFILE,
        'membrane_id': MEMB_ID,
        'surface_fid_override': FID_OVERRIDE,
        'surface_profile_subset': SURFACE_PROFILE_SUBSET,
        'surface_include_hole': INCLUDE_THROUGH_HOLE,
        'surface_relative_permittivity_dielectric': SURFACE_DIELECTRIC_RELATIVE_PERMITTIVITY,
        'depth': DEPTH,
        'radius': RADIUS,
        'dict_settings_radius': DICT_SETTINGS_RADIUS,
        'tck_smoothing': SMOOTHING,
        'max_dz_for_strain_plot': MAX_DZ_FOR_STRAIN_PLOT,
        'memb_original_thickness': MEMB_ORIGINAL_THICKNESS,
        'memb_original_youngs_modulus': MEMB_ORIGINAL_YOUNGS_MODULUS,
        'experimental_pre_stretch_nominal': EXPERIMENTAL_PRE_STRETCH_NOMINAL,
        'experimental_pre_stretch_measured': EXPERIMENTAL_PRE_STRETCH_MEASURED,
        'memb_thickness_post_measured_pre_stretch': MEMB_THICKNESS_POST_MEASURED_PRE_STRETCH,
        'gent_model_computed_residual_stress_from_pre_stretch_measured': GENT_MODEL_COMPUTED_RESIDUAL_STRESS_FROM_PRE_STRETCH_MEASURED,
        'model_use_pre_metal_pre_stretch_for_membrane_thickness': MODEL_USE_PRE_METAL_PRE_STRETCH_FOR_MEMB_THICKNESS,
        'memb_youngs_modulus_bulge_test': MEMB_YOUNGS_MODULUS_BULGE_TEST,
        'memb_residual_stress_bulge_test': MEMB_RESIDUAL_STRESS_BULGE_TEST,
        'gent_model_computed_pre_stretch_from_residual_stress_bulge_test': GENT_MODEL_COMPUTED_PRE_STRETCH_FROM_RESIDUAL_STRESS_BULGE_TEST,
        'gent_model_j_memb': MEMB_ORIGINAL_GENT_MODEL_J_MEMB,
        'include_metal_energy': INCLUDE_METAL_ENERGY,
        'metal_layers': METAL_LAYERS,
        'youngs_modulus_metal_GPa': E_METAL,
        'poissons_ratio_metal': NU_METAL,
        'thickness_metal_nm': THICKNESS_METAL,
        'units': UNITS,
        'model_num_segments': NUM_SEGMENTS,
        'model_config': config,
        'model_shape': shape,
        'model_diameter_um': diameter,
        'model_depth_um': depth,
        'model_thickness_membrane_um': t*1e6,
        'model_pre_stretch': pre_stretch,
        'model_thickness_dielectric_um': t_diel,
        'model_youngs_modulus': Youngs_Modulus,
        'model_J_memb': J_memb,
        'model_eps_r_memb': eps_r_memb,
        'model_eps_r_diel': eps_r_diel,
        'model_surface_roughness_diel': surface_roughness_diel,
        'model_Vmin': U_is.min(),
        'model_Vmax': U_is.max(),
        'model_Vstep': U_is[1] - U_is[0],
    }
    DF_MODEL_SETTINGS = pd.DataFrame.from_dict(data=DICT_MODEL_SETTINGS, orient='index', columns=['v'])
    DF_MODEL_SETTINGS.to_excel(join(SAVE_DIR, SAVE_ID + '_model_settings.xlsx'), index=True, index_label='k')

    # profile to use
    px, py = dict_fid['r'], dict_fid['z']
    if not os.path.exists(join(SAVE_DIR, SAVE_ID + '_profile.png')):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(px * 1e3, py * 1e6, label=np.round(np.min(py * 1e6), 1))
        ax.set_xlabel('mm')
        ax.set_ylabel('um')
        ax.grid(alpha=0.25)
        ax.legend(title='Depth (um)')
        ax.set_title('dz/segment = {} um ({} segs)'.format(np.round(np.min(py * 1e6) / len(px), 2), len(px)))
        plt.suptitle(SAVE_ID)
        plt.tight_layout()
        plt.savefig(join(SAVE_DIR, SAVE_ID + '_profile.png'), facecolor='w')
        plt.close()
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
    dict_metal = {
        'E': youngs_modulus_metal,
        'poissons_ratio': poissons_ratio_metal,
        'thickness': t_metal,
        'include_metal': include_metal_energy,
    }
    # ----------

    #   SOLVE THE MODEL

    # ----------
    dZ_clip = 2.5

    save_mechanical = join(SAVE_DIR, 'mechanical')
    save_electrical = join(SAVE_DIR, 'electrical')
    save_energy_minima = join(SAVE_DIR, 'energy_minima')
    for pth in [save_mechanical, save_electrical, save_energy_minima]:
        if not os.path.exists(pth):
            os.makedirs(pth)

    # --- MECHANICAL ENERGY ONLY
    df = calculate_elastic_energy_of_zipped_profile(config, dict_actuator, dict_material, U_is=[0], dict_metal=dict_metal)
    df.to_excel(join(save_mechanical, SAVE_ID + '_model_elastic_energy.xlsx'))
    plot_energy_by_depth(df, save_mechanical, save_id=SAVE_ID + '_model_elastic_energy_vs_depth.png', z_clip=dZ_clip)

    df = calculate_strain_by_z(df)
    df.to_excel(join(SAVE_DIR, SAVE_ID + '_model_strain-by-z.xlsx'))
    DZ_LESS_THAN_FOR_STRAIN_PLOT = 1
    plot_deformation_by_depth(df, SAVE_DIR, SAVE_ID + '_model_strain-by-z.png')
    # df['added_stretch'] = df['stretch_f'] - df['stretch_i'].iloc[0]
    # df['disp_r_microns'] = df['added_stretch'] * df['x_f'] / 2 * 1e6  # divide by 2 = radial displacement

    # --- MECHANICAL + ELECTROSTATIC ENERGY
    # for several voltages
    df_U_is = calculate_elastic_energy_of_zipped_profile(config, dict_actuator, dict_material, U_is=U_is, dict_metal=dict_metal)
    cols_df_U_is = ['U_i', 'step', 'dL', 'dX', 'dZ', 'L_i', 'x_i', 't_i', 'stretch_i', 'perimeter_i', 'sa_i', 'vol_i',
                   'Em_seg_i', 'Es_seg_i', 'L0_f', 'L_f', 'x_f', 't_f', 'stretch_f', 'vol_f', 'Em_flat_i',
                   'Em_seg_sum_i', 'Em_tot_i', 'Es_seg_sum_i', 'E_tot_i']
    df_U_is = df_U_is[['U_i', 'dZ', 'x_f', 't_f', 'stretch_f', 'Em_seg_i', 'Es_seg_i', 'Em_flat_i',
                       'Em_seg_sum_i', 'Em_tot_i', 'Es_seg_sum_i', 'E_tot_i']]
    df_U_is.to_excel(join(save_electrical, SAVE_ID + '_model_total_energy.xlsx'))


    # ---

    # calculate some set up parameters
    dZ_pull_in = np.min(py)
    dZ_max_plot = dZ_pull_in + dZ_clip * 1e-6
    dZ_i = 0
    dZ_is = []

    min_allowable_pull_in_voltage = 80
    dZ_trigger_min_allowable_pull_in_voltage = dZ_pull_in + 10e-6

    for voltage in U_is:
        df = df_U_is[df_U_is['U_i'] == voltage]
        # Find the first energy minima
        px, py4 = 'dZ', 'E_tot_i'
        x, y = df[px].to_numpy(), df[py4].to_numpy()
        first_minima_x, first_minima_y = find_first_energy_minima(x, y, print_id=f'({voltage}V)')

        if not np.isnan(first_minima_x):
            if (
                    dZ_i == 0 and                                               # if previous stable position was z=0
                    voltage < min_allowable_pull_in_voltage and                 # if voltage is "low"
                    first_minima_x > dZ_trigger_min_allowable_pull_in_voltage   # if first minima occurs near pull-in
            ):
                dZ_i = 0
            else:
                dZ_i = first_minima_x

        dZ_is.append(dZ_i)

        if voltage % 25 != 0 or dZ_i < dZ_max_plot:
            continue

        plot_energy_by_depth(df, save_electrical,
                             save_id=SAVE_ID + '_model_total_energy_vs_depth_{}V.png'.format(voltage))


        path_save = save_energy_minima
        save_id = SAVE_ID + '_first_energy_minima_{}V.png'.format(voltage)

        # plot elastic energy vs. depth
        x = x * 1e6
        py1a, py1b = 'Em_seg_i', 'Es_seg_i'
        py2a, py2b = 'Em_seg_sum_i', 'Em_flat_i'
        py3a, py3b = 'Em_tot_i', 'Es_seg_sum_i'

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(6, 8), sharex=True)
        # Energy of segments
        ax1.plot(x, df[py1a], label=py1a)
        ax1.plot(x, df[py1b], label=py1b)
        # (Mechanical) energy of zipped and flat membrane
        ax2.plot(x, df[py2a], label=py2a)
        ax2.plot(x, df[py2b], label=py2b)
        # Total electrostatic and mechanical energy
        ax3.plot(x, df[py3a], label=py3a)
        ax3.plot(x, df[py3b], label=py3b)
        # Total energy
        ax4.plot(x, df[py4], label=py4)
        if not np.isnan(first_minima_x):
            # first_minima_y
            ax4.scatter(first_minima_x*1e6, first_minima_y, s=10, marker='*', color='k',
                        label='First minima: {} um'.format(np.round(first_minima_x * 1e6, 1)))
            # ax4.axvline(first_minima_x*1e6, color='k', linestyle='--', lw=0.7, label='First minima: {} um'.format(np.round(first_minima_x * 1e6, 1)))

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_ylabel('Energy (J)')
            ax.grid(alpha=0.25)
            ax.legend(fontsize='x-small')
        ax4.set_xlabel('Depth (um)')
        plt.tight_layout()
        plt.savefig(join(path_save, save_id), dpi=300, facecolor='w', bbox_inches='tight')
        plt.close()

    df_dZ = pd.DataFrame(np.vstack([U_is, dZ_is]).T, columns=['U', 'z'])
    df_dZ['E'] = Youngs_Modulus
    df_dZ['pre_stretch'] = pre_stretch
    df_dZ.to_excel(join(SAVE_DIR, SAVE_ID + '_model_z-by-v.xlsx'))

    px = 'U'
    py = 'z'
    x = df_dZ[px].to_numpy()
    y = df_dZ[py].to_numpy()

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x, y * 1e6, label='dZ')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Depth (um)')
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(join(SAVE_DIR, SAVE_ID + '_model_z-by-v.png'),
                facecolor='w', dpi=300, bbox_inches='tight')