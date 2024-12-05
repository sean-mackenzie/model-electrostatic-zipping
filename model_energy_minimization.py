# tests/test_model_sweep.py

# imports
import numpy as np
import pandas as pd

from utils.shapes import surface_area, perimeter
from utils.energy import mechanical_energy_density_Gent, electrostatic_energy_density_SR


def solve_energy_iterative_shape_function(config, dict_actuator, dict_material,
                                          U_is, append_dfs=False, export_excel=False,
                                          save_id='arb'):
    """

    :param config:
    :param dict_actuator:
    :param dict_material:
    :param num_segments:
    :param U_is:
    :param append_dfs:
    :param export_excel:
    :param save_id:
    :param silence:
    :return:
    """

    actuator_shape = dict_actuator['shape']
    X = dict_actuator['diameter']
    Z = dict_actuator['depth']
    profile_x = dict_actuator['profile_x']
    profile_z = dict_actuator['profile_z']

    # --- CHANGES:
    #X = dict_actuator['diameter']  # 0.0015 dict_actuator['diameter'] --> useful only if want to artificially add dia_flat to given profiles
    #Z = dict_actuator['depth']  # 0.0002 dict_actuator['depth'] --> practically not used at all
    # ---

    # --- THINGS THAT COULD BE PUT INTO A "DICT_MEMBRANE"
    t = dict_actuator['membrane_thickness']
    pre_stretch = dict_actuator['pre_stretch']

    t_diel = dict_actuator['dielectric_thickness']
    eps_r_diel = dict_material['eps_r_diel']
    surface_roughness = dict_material['surface_roughness_diel']

    if 'E' in dict_material.keys():
        mu = dict_material['E'] / 3
    elif 'mu' in dict_material.keys():
        mu = dict_material['mu']
    else:
        raise ValueError("Must provide either 'E' (Youngs modulus) or 'mu' (shear modulus).")
    J = dict_material['Jm']
    eps_r_memb = dict_material['eps_r_memb']

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
        except np.linalg.LinAlgError:
            import matplotlib.pyplot as plt
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


def legacy_solve_energy_iterative_shape_function(config, dict_actuator, dict_material,
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

        test_plot = False
        if test_plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(pxn, py3n, 'k-')
            ax.plot(pxn, pf12(pxn), 'r--')
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