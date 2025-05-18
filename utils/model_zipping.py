
# imports
import os
from os.path import join
import numpy as np
import pandas as pd

from utils.shapes import surface_area, perimeter
from utils.energy import mechanical_energy_density_Gent, mechanical_energy_density_NeoHookean, electrostatic_energy_density_SR
from  utils.energy import mechanical_energy_density_metal, mechanical_energy_density_metal_3D
from utils import empirical, settings, materials
from utils.empirical import dict_from_tck
from tests.test_manually_fit_tck_to_surface_profile import manually_fit_tck

import matplotlib.pyplot as plt
# Customize the property cycle to include both colors and line styles
from cycler import cycler
linestyles = ['-', '--', '-.', ':']
# default_colors = ['r', 'g', 'b', 'y']
# default_cycler = (cycler(color=default_colors) + cycler(linestyle=linestyles))
use_colors = ['k', 'r', 'b', 'g']  # , 'm', 'c', 'y'
use_cycler = (cycler(color=use_colors) + cycler(linestyle=linestyles))
plt.rc('axes', prop_cycle=use_cycler)
# for reference:
# cmap_colors = plt.cm.plasma([0.1, 0.4, 0.7, 1.0])  # Use a colormap to generate colors
# colormap_cycler = (cycler('color', cmap_colors) + cycler('linestyle', linestyles))
# plt.rc('axes', prop_cycle=colormap_cycler)




def calculate_zipped_segment_volumes_by_depth(dict_geometry):
    """
    Calculates the segmented volumes of a surface profile by its depth, accounting for the properties
    of a membrane and a metal film. This function processes the geometry of a zipped actuator surface
    profile, computes volumes, and mechanical properties of each zipped segment accordingly. The final
    result is returned as a DataFrame containing values for both zipped and flat segments.

    outputs_volumes = [
        'step',
        'dL', 'dX', 'dZ', 'x_i', 'perimeter_i', 'sa_i',
        't_i', 'vol_i', 'stretch_i',
        't_i_metal', 'vol_i_metal', 'stretch_i_metal',
        'x_flat', 'sa_flat',
        't_flat', 'vol_flat', 'stretch_flat',
        't_flat_metal', 'vol_flat_metal', 'stretch_flat_metal',
    ]

    :param dict_geometry: A dictionary that defines the geometric and material properties
                          of the surface profile and membrane. It contains:
                          - `shape`: Shape type of the profile.
                          - `profile_x`: 1D numpy array representing x-coordinates of the profile.
                          - `profile_z`: 1D numpy array representing z-coordinates of the profile.
                          - `memb_t`: Thickness of the membrane (float).
                          - `memb_ps`: Pre-stretch factor for the membrane (float).
                          - `met_t`: Thickness of the metal film (float).
    :return: A pandas DataFrame where rows correspond to simulation steps, and columns hold calculated
             properties for zipped segments and flat segments.
    :rtype: pandas.DataFrame
    """
    # Surface Profile
    actuator_shape = dict_geometry['shape']
    profile_x = dict_geometry['profile_x']
    profile_z = dict_geometry['profile_z']
    X = dict_geometry['radius'] * 2  # profile_x.max() * 2

    # Membrane or Composite
    t = dict_geometry['memb_t']
    pre_stretch = dict_geometry['memb_ps']
    t0 = t * pre_stretch ** 2  # dict_geometry['memb_t0'] # You must define t0 this way, otherwise won't work

    ### If you want to vary pre-stretch, while assuming
    ### a constant t0, then you need to enable below.
    ### Otherwise, the original thickness t0 will vary.
    # t0 = dict_geometry['memb_t']
    # t = t / pre_stretch ** 2

    print("Membrane original thickness, t0: ", t0)
    print("Membrane thickness at start of zipping, t: ", t)

    # Metal
    t_metal = dict_geometry['met_t']  # meters
    t_ps_metal = dict_geometry['met_ps']

    # ---

    # --- SOLVER

    dL_i = 0  # dL_i: current total length (hypotenuse) of zipped segments
    dX_i = 0  # dX_i: current x-direction length of zipped segments
    dZ_i = 0  # dZ_i (z_i): current z-direction length of zipped segments

    x_i = X  # (X_i) moving width of surface profile (effectively, diameter)
    t_i = t  # moving membrane thickness
    Vol_i = surface_area(l=x_i, shape=actuator_shape) * t_i  # moving volume of flat membrane
    stretch_i = pre_stretch  # moving stretch in flat membrane
    t_i_metal = t_metal  # moving metal thickness
    Vol_i_metal = surface_area(l=x_i, shape=actuator_shape) * t_i_metal  # moving volume of flat metal film
    stretch_i_metal = t_ps_metal  # moving pre-stretch of the metal film

    res = []
    for i in np.arange(1, len(profile_x)):
        # --- CALCULATE VALUES FOR ZIPPED SEGMENT OF MEMBRANE
        # 1. evaluate moving positions
        dX = profile_x[i] - profile_x[i - 1]
        dZ = (profile_z[i] - profile_z[i - 1]) * -1
        dL = np.sqrt(dX ** 2 + dZ ** 2)
        slope = dZ / dX
        slope_angle = np.rad2deg(np.arctan(slope))
        dX_i += dX  # dX_i: current x-direction length of zipped segments
        dZ_i += dZ  # dZ_i (z_i): current z-direction length of zipped segments
        dL_i += dL  # dL_i: current total length (hypotenuse) of zipped segments

        # ---

        # 2. evaluate current zipped segment
        perimeter_i = perimeter(l=x_i, shape=actuator_shape)  # circumference of profile at this position
        """ THIS IS BELIEVED TO BE WRONG: as of 4/14/2025
        perimeter_i = perimeter(l=x_i - dX / 2, shape=actuator_shape)   # circumference of profile at this position
        surface_area_i = dL * perimeter_i                               # surface area of zipped segment
        vol_i = surface_area_i * t_i                                    # volume of zipped segment
        vol_i_metal = surface_area_i * t_i_metal                        # volume of zipped metal segment
        """
        surface_area_of_flat_membrane_before_zipping = surface_area(l=x_i, shape=actuator_shape)
        surface_area_of_flat_membrane_less_zipped_segment = surface_area(l=x_i - 2 * dL, shape=actuator_shape)

        # The difference in area must be equal to that of the zipped segment, regardless of its orientation
        surface_area_i = surface_area_of_flat_membrane_before_zipping - surface_area_of_flat_membrane_less_zipped_segment

        vol_i = surface_area_i * t_i  # volume of zipped segment
        vol_i_metal = surface_area_i * t_i_metal  # volume of zipped metal segment

        # Data formulating the zipped segment
        res_zip_i = [i, dL, dX, dZ, slope, slope_angle,
                     dL_i, dX_i, dZ_i, x_i, perimeter_i, surface_area_i, # purely geometric
                     t_i, vol_i, stretch_i,  # geometry of this zipped membrane segment
                     t_i_metal, vol_i_metal, stretch_i_metal,  # geometry of this zipped metal segment
                     ]
        # ---
        # --- CALCULATE THE NEW VALUES FOR SUSPENDED MEMBRANE
        # 2. evaluate new flat membrane
        # geometry
        x_i = x_i - 2 * dX  # new width of surface profile
        surface_area_f = surface_area(l=x_i, shape=actuator_shape)
        Vol_i = Vol_i - vol_i  # new volume of suspended membrane
        if Vol_i < 0:
            continue
        t_i = Vol_i / surface_area_f  # new thickness of suspended membrane
        stretch_i = np.sqrt(t0 / t_i)  # new stretch of suspended membrane
        # metal
        Vol_i_metal = Vol_i_metal - vol_i_metal  # new volume of metal film on suspended membrane
        t_i_metal = Vol_i_metal / surface_area_f  # new thickness of metal on suspended membrane
        # The below doesn't add the additional stretch to the original pre-stretch
        # stretch_i_metal = np.sqrt(t_metal / t_i_metal)  # new stretch of metal on suspended membrane
        # The below should add the additinal stretch to the original pre-stretch
        stretch_i_metal = np.sqrt(t_metal / t_i_metal) - 1 + t_ps_metal  # new stretch of metal on suspended membrane
        # Data formulating the flat membrane
        res_flat_i = [x_i, surface_area_f, t_i, Vol_i, stretch_i, t_i_metal, Vol_i_metal, stretch_i_metal]
        # ---
        # Add all results and append to list
        res_i = res_zip_i + res_flat_i
        res.append(res_i)

    # dataframe
    columns_zip = ['step', 'dL_i', 'dX_i', 'dZ_i', 'slope_i', 'theta_i',
                   'dL', 'dX', 'dZ', 'x_i', 'perimeter_i', 'sa_i',
                   't_i', 'vol_i', 'stretch_i',
                   't_i_metal', 'vol_i_metal', 'stretch_i_metal',
                   ]
    columns_flat = ['x_flat', 'sa_flat',
                    't_flat', 'vol_flat', 'stretch_flat',
                    't_flat_metal', 'vol_flat_metal', 'stretch_flat_metal']
    columns = columns_zip + columns_flat
    df = pd.DataFrame(np.vstack(res), columns=columns)

    # --- Incompressibility requirement (conservation of mass)
    df['vol_seg_sum_i'] = df['vol_i'].cumsum()
    df['vol_conservation_i'] = df['vol_seg_sum_i'] + df['vol_flat']

    return df


def calculate_radial_displacement_by_depth(df, poly_deg, path_save=None, save_id=''):
    df['stretch_due_to_zipping'] = np.sqrt(df['t_i'].iloc[0] / df['t_flat'])

    # NOTE, 5/16/2025: I just realized that the correct way to calculate radial displacement is:
    # dr = stretch * r0, where r0 is the initial radial position of the infinitesimal membrane segment.
    # So, the radial displacement that I've been calculating is actually slightly larger than it should be.
    # However, the error associated with this mistake scales with dr relative to r, so, for my scenarios,
    # where dr is < 5 microns and r is > 500 microns, this error is negligible. Nevertheless, it should
    # be corrected. The correct calculation of radial displacement is: dr = r(lambda - 1) / lambda
    df['disp_r_microns'] = ((df['x_flat'] / 2 * 1e6) * (df['stretch_due_to_zipping'] - 1)) / df['stretch_due_to_zipping']

    # For reference (i.e., to evaluate the error of my mistake), I will keep the mistaken radial displacement
    df['disp_r_microns_wrong'] = (df['stretch_due_to_zipping'] - 1) * df['x_flat'] / 2 * 1e6  # divide by 2 = radial displacement

    df['apparent_dr_microns_discrete'] = np.sin(np.deg2rad(df['theta_i'])) * df['t_i'] * -1e6
    df['disp_r_microns_corr_discrete'] = df['disp_r_microns'] + df['apparent_dr_microns_discrete']

    # The "old" (before 4/14/25) method of calculating these are below. They give identical results
    # But I feel more comfortable using the above approach, which makes more intuitive sense.
    # df['stretch_ratio'] = df['stretch_flat'] / df['stretch_i'].iloc[0]
    # df['disp_r_microns'] = (df['stretch_ratio'] - 1) * df['x_flat'] / 2 * 1e6  # divide by 2 = radial displacement

    # "new" (5/2/2025 and after)
    df = calculate_apparent_radial_displacement_including_thickness(df, poly_deg, path_save, save_id)

    return df


def calculate_apparent_radial_displacement_including_thickness(df, poly_deg, path_save=None, save_id=''):
    x, y = df['dX'].to_numpy(), df['dZ'].to_numpy()
    # Clip surface profile data to only include the sloped portion
    #x = x[y < z_clip]
    #y = y[y < z_clip]
    # Fit polynomial to surface profile
    coefficients = np.polyfit(x, y, deg=poly_deg)
    polynomial = np.poly1d(coefficients)
    # Derivative of the fitted polynomial is smoothed surface slope
    polynomial_derivative = polynomial.deriv()
    df['slope_i_smooth'] = polynomial_derivative(x)

    # inverse tangent of slope gives sidewall angle
    angle_radians = np.arctan(polynomial_derivative(x))
    angle_degrees = np.rad2deg(angle_radians)
    df['theta_i_smooth'] = angle_degrees

    # Apparent radial displacement = membrane thickness + slope angle
    apparent_displacement = np.sin(angle_radians) * df['t_i'].to_numpy() * -1  # minus sign implies opposite of stretch
    # there should never be apparent radial displacements in the opposite direction, so replace with 0 if there are
    apparent_displacement = np.where(apparent_displacement > 0, 0, apparent_displacement)
    df['apparent_dr_microns'] = apparent_displacement * 1e6  # convert units to microns

    df.loc[df['dZ'] < 0.1e-6, 'apparent_dr_microns'] = 0
    # df.loc[df['dZ'] < df['dZ'].min() + df['t_f'].iloc[-1] / 4, 'apparent_dr_microns'] = 0

    df['disp_r_microns_corr'] = df['disp_r_microns'] + df['apparent_dr_microns']

    # plot fitted surface profile to double-check fit is good
    plot_polynomial_fitted_to_surface_profile(df, x, y, polynomial, polynomial_derivative, poly_deg, path_save, save_id)

    return df


def plot_polynomial_fitted_to_surface_profile(df, x, y, polynomial, polynomial_derivative, poly_deg, path_save=None, save_id=''):
    # convert data to units/direction that makes intuitive sense with surface profile
    x_um, y_um = x * 1e6, y * -1e6
    # get radial min/max position where the correction should actually be applied
    """
    memb_thickness_at_bottom = 10
    relative_thicknesses = 0.25
    tilt_corr_only_up_to = np.min(y_um) + memb_thickness_at_bottom * relative_thicknesses
    x_applicable = x_um[(y_um < -0.1) & (y_um > tilt_corr_only_up_to)]
    """

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(5, 7.25),
                                             gridspec_kw={'height_ratios': [1, 1, 1, 1.35]})  # (7, 8)
    # ax1.scatter(x, y, s=4, alpha=0.95, label='Surface profile')
    ax1.plot(x_um, y_um, 'k-', label='Surface profile')
    ax1.plot(x_um, polynomial(x) * -1e6, 'r-', lw=0.85, label=f'{poly_deg}-deg Polynominal')
    ax2.plot(x_um, polynomial_derivative(x), label='Derivative (Slope)', color='green', linestyle='--')
    ax3.plot(x_um, df['theta_i_smooth'], label='Angle (degrees)', color='purple')
    # -
    ax4.plot(x_um, df['disp_r_microns'], 'k-', label=r'$\Delta r_{Bot}$',)
    ax4.plot(x_um, df['disp_r_microns_wrong'], 'b--', lw=0.5, label=r'$\Delta r_{Bot, wrong}$',)
    ax4.plot(x_um, df['apparent_dr_microns'], 'r-', label=r'$\Delta r_{Tilt}$',)
    ax4.plot(x_um, df['disp_r_microns_corr'], 'g-', label=r'$\Delta r_{Top}$',)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(alpha=0.15)
        #ax.axvline(x_applicable.min(), color='k', lw=0.5, linestyle='--', alpha=0.15)
        #ax.axvline(x_applicable.max(), color='k', lw=0.5, linestyle='--', alpha=0.15)
    """
    ax1.axhline(y=tilt_corr_only_up_to, color='k', lw=0.5, linestyle='--', alpha=0.15)
    ax4.axvline(x_applicable.min(), color='k', lw=0.5, linestyle='--', alpha=0.15,
                label='rmin={}'.format(np.round(x_applicable.min(), 1)))
    ax4.axvline(x_applicable.max(), color='k', lw=0.5, linestyle='--', alpha=0.15,
                label='rmax(t={}um*{})={}'.format(memb_thickness_at_bottom, relative_thicknesses,
                                                  np.round(x_applicable.max(), 1)))
    """

    ax1.legend(fontsize='x-small')
    ax2.legend(fontsize='x-small')
    ax4.legend(fontsize='xx-small')
    ax1.set_ylabel(r'$z \: (\mu m)$')
    ax2.set_ylabel('slope')
    ax3.set_ylabel('angle (degrees)')
    ax4.set_ylabel(r'$\Delta r \: (\mu m)$')
    ax4.set_xlabel(r'$r \: (\mu m)$')
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, f'apparent_radial_displacement_{save_id}.png'),
                    dpi=300, facecolor='w', bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def get_surface_slope_function(df, poly_deg):
    # Fit polynomial to surface profile
    coefficients = np.polyfit(df['dX'].to_numpy(), df['dZ'].to_numpy(), deg=poly_deg)
    polynomial = np.poly1d(coefficients)
    # Derivative of the fitted polynomial
    polynomial_derivative = polynomial.deriv()
    return polynomial_derivative


def plot_deformation_by_depth(df, path_save, save_id, z_clip=2.5):
    px = 'dZ'
    py1 = 't_flat'
    py2, lbl2 = 'stretch_flat', r'$\lambda_{total}$'
    py2b, lbl2b = 'stretch_due_to_zipping', r'$\lambda_{zipping}$'
    py3 = 'disp_r_microns'
    py3_corr = 'apparent_dr_microns'
    py3b = 'disp_r_microns_corr'

    x3 = df[px].to_numpy() * 1e6
    y3 = df[py3].to_numpy()
    y3_corr = df[py3_corr].to_numpy()
    y3b = df[py3b].to_numpy()

    df = df[df[px] < df[px].max() - z_clip * 1e-6]
    x12 = df[px] * 1e6
    y1 = df[py1] * 1e6
    y2 = df[py2]
    y2b = df[py2b]

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(4, 6), sharex=True)

    ax1.plot(x12, y1, label=py1)
    ax1.set_ylabel(r'$t_{memb} \: (\mu m)$')
    ax1.grid(alpha=0.25)

    ax2.plot(x12, y2, label=lbl2)
    ax2.plot(x12, y2b, label=lbl2b)
    ax2.set_ylabel(r'$\lambda$')
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize='small')

    ax3.plot(x3, y3, label=r'$\Delta r$')
    ax3.plot(x3, y3b, label=r'$\Delta r_{corr}$')
    ax3.set_ylabel(r'$\Delta r_{o, zipping} \: (\mu m)$')
    ax3.grid(alpha=0.25)
    ax3.legend(fontsize='small')
    ax3.set_xlabel('Depth (um)')

    plt.tight_layout()
    plt.savefig(join(path_save, save_id), dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()


def calculate_elastic_energy_by_depth(df, dict_mechanical, use_neo_hookean=False):
    """
    Calculates the elastic energy of membrane and metal components of a composite
    material as functions of depth (dZ). The function adds several calculated
    columns to the input dataframe (`df`), representing the membrane and metal
    elastic energy densities and their sums. These calculations are performed for
    zipped segments, flat membrane states, and total states (comprising both
    zipped and flat states).

    The input mechanical properties are provided in the dictionary
    `dict_mechanical`, which includes the relevant parameters for calculating the
    mechanical energy densities of the membrane and metal components.

    inputs_elastic = [
        'vol_i', 'stretch_i',
        'vol_i_metal', 'stretch_i_metal',
        'vol_flat', 'stretch_flat',
        'vol_flat_metal', 'stretch_flat_metal',
    ]
    outputs_elastic = [
        'Em_seg_i_memb', 'Em_seg_i_metal',
        'Em_seg_i_mm', 'Em_seg_i_comp',
        'Em_seg_sum_i_memb', 'Em_seg_sum_i_metal', 'Em_seg_sum_i_mm', 'Em_seg_sum_i_comp',
        'Em_flat_i_memb', 'Em_flat_i_metal', 'Em_flat_i_mm', 'Em_flat_i_comp',
        'Em_tot_i_memb', 'Em_tot_i_metal', 'Em_tot_i_mm', 'Em_tot_i_comp',
    ]

    :param df: Input dataframe, which should contain columns representing volumes
        of the segments (`vol_i`, `vol_i_metal`, `vol_flat`, `vol_flat_metal`),
        stretch ratios (`stretch_i`, `stretch_i_metal`, `stretch_flat`,
        `stretch_flat_metal`), and other associated physical properties.
    :param dict_mechanical: Dictionary containing mechanical properties relevant
        to the energy calculations. Expected keys include 'memb_E'
        (membrane elastic modulus), 'memb_J' (deformation parameter for Gent
        model), and 'met_E' and 'met_nu' (Young's modulus and Poisson's ratio,
        respectively, for metal components).
    :return: DataFrame containing the input data along with additional columns
        calculated for the elastic energy of the membrane, metal, and composite
        components.

    """
    memb_E = dict_mechanical['memb_E']
    memb_J = dict_mechanical['memb_J']
    met_E = dict_mechanical['met_E']
    met_nu = dict_mechanical['met_nu']
    comp_E = dict_mechanical['comp_E']

    # --- Elastic energy

    # Elastic energy of zipped segment at each dZ
    # Membrane (E = 1.2 MPa)
    if use_neo_hookean:
        df['Em_seg_i_memb'] = df['vol_i'] * mechanical_energy_density_NeoHookean(mu=memb_E / 3, l=df['stretch_i'])
    else:
        df['Em_seg_i_memb'] = df['vol_i'] * mechanical_energy_density_Gent(mu=memb_E / 3, J=memb_J, l=df['stretch_i'])

    # Metal (Young's modulus derived from composite bulge test)
    # df['Em_seg_i_metal'] = df['vol_i_metal'] * mechanical_energy_density_metal(E=met_E, nu=met_nu, l=df['stretch_i_metal'])
    df['Em_seg_i_metal'] = df['vol_i_metal'] * mechanical_energy_density_metal_3D(E=met_E, nu=met_nu, l=df['stretch_i_metal'])

    # Total energy of membrane + metal
    df['Em_seg_i_mm'] = df['Em_seg_i_memb'] + df['Em_seg_i_metal']
    # Composite membrane bilayer (Young's modulus from bulge test)
    if use_neo_hookean:
        df['Em_seg_i_comp'] = df['vol_i'] * mechanical_energy_density_NeoHookean(mu=comp_E / 3,l=df['stretch_i'])
    else:
        df['Em_seg_i_comp'] = df['vol_i'] * mechanical_energy_density_Gent(mu=comp_E / 3, J=memb_J, l=df['stretch_i'])

    # Total elastic energy of all zipped segments as a function of dZ (sum: 0 to dZ)
    df['Em_seg_sum_i_memb'] = df['Em_seg_i_memb'].cumsum()
    df['Em_seg_sum_i_metal'] = df['Em_seg_i_metal'].cumsum()
    df['Em_seg_sum_i_mm'] = df['Em_seg_i_mm'].cumsum()
    df['Em_seg_sum_i_comp'] = df['Em_seg_i_comp'].cumsum()

    # Elastic energy of flat membrane at each dZ
    if use_neo_hookean:
        df['Em_flat_i_memb'] = df['vol_flat'] * mechanical_energy_density_NeoHookean(mu=memb_E / 3, l=df['stretch_flat'])
    else:
        df['Em_flat_i_memb'] = df['vol_flat'] * mechanical_energy_density_Gent(mu=memb_E / 3, J=memb_J, l=df['stretch_flat'])
    df['Em_flat_i_metal'] = df['vol_flat_metal'] * mechanical_energy_density_metal(E=met_E, nu=met_nu, l=df['stretch_flat_metal'])
    df['Em_flat_i_mm'] = df['Em_flat_i_memb'] + df['Em_flat_i_metal']
    if use_neo_hookean:
        df['Em_flat_i_comp'] = df['vol_flat'] * mechanical_energy_density_NeoHookean(mu=comp_E / 3, l=df['stretch_flat'])
    else:
        df['Em_flat_i_comp'] = df['vol_flat'] * mechanical_energy_density_Gent(mu=comp_E / 3, J=memb_J, l=df['stretch_flat'])

    # Total elastic energy of zipped + flat membrane at each dZ
    df['Em_tot_i_memb'] = df['Em_seg_sum_i_memb'] + df['Em_flat_i_memb']
    df['Em_tot_i_metal'] = df['Em_seg_sum_i_metal'] + df['Em_flat_i_metal']
    df['Em_tot_i_mm'] = df['Em_seg_sum_i_mm'] + df['Em_flat_i_mm']
    df['Em_tot_i_comp'] = df['Em_seg_sum_i_comp'] + df['Em_flat_i_comp']

    return df


def calculate_electrostatic_energy_by_depth_for_voltage(df, dict_electrostatic, voltage):
    """
    Calculates the electrostatic energy for a given depth and voltage. This function
    computes the energy density for each segment within the depth based on the
    provided parameters and sums the energy density across segments for a cumulative
    electrostatic energy calculation.

    inputs_electrostatic = [
        'sa_i'
    ]
    outputs_electrostatic = [
        'Es_seg_i', 'Es_seg_sum_i',
    ]

    :param df: A pandas DataFrame containing the necessary columns for surface area
        and depth to calculate electrostatic energy. The DataFrame is also updated
        with new calculated columns for segment energy and cumulative energy.
    :type df: pandas.DataFrame
    :param dict_electrostatic: A dictionary containing parameters required for
        electrostatic energy calculations, including:
            - sd_eps_r (relative permittivity)
            - sd_t (thickness)
            - sd_sr (surface area radius)
    :param voltage: The applied voltage for the electrostatic energy calculation.
    :return: A pandas DataFrame with additional columns:
        - 'Es_seg_i' representing the electrostatic energy density for each segment
          at the corresponding depth.
        - 'Es_seg_sum_i' representing the cumulative electrostatic energy across all
          segments as a function of depth.
    :rtype: pandas.DataFrame
    """
    sd_eps_r = dict_electrostatic['sd_eps_r']
    sd_t = dict_electrostatic['sd_t']
    sd_sr = dict_electrostatic['sd_sr']

    # Electrostatic energy of zipped segment at each dZ
    df['Es_seg_i'] = electrostatic_energy_density_SR(eps_r=sd_eps_r, A=df['sa_i'], d=sd_t, U=voltage, R0=sd_sr)
    # Total electrostatic energy of all zipped segments as a function of dZ (sum: 0 to dZ)
    df['Es_seg_sum_i'] = df['Es_seg_i'].cumsum()

    return df


def calculate_total_energy_by_depth(df):
    """
    Calculate the total energy of the system by depth.

    This function computes the total energy for different components of
    a system using the columns present in the given DataFrame. Specifically,
    it calculates the total energy at the membrane, membrane-matrix, and
    composite layers by summing relevant columns within the DataFrame.

    input:
        df = df_elastic[columns_id + outputs_elastic].join(df_electrostatic[outputs_electrostatic])

    outputs_total_energy = [
        'E_tot_i_memb', 'E_tot_i_mm', 'E_tot_i_comp',
    ]

    :param df: Input DataFrame containing the required energy components
        - Em_tot_i_memb: Total energy for membrane initialization
        - Es_seg_sum_i: Segment energy sum for each layer
        - Em_tot_i_mm: Total energy for membrane-matrix initialization
        - Em_tot_i_comp: Total energy for composite initialization
    :type df: pandas.DataFrame
    :return: Updated DataFrame with additional columns for total energy
        - E_tot_i_memb: Total energy for the membrane
        - E_tot_i_mm: Total energy for the membrane-matrix
        - E_tot_i_comp: Total energy for the composite layer
    :rtype: pandas.DataFrame
    """
    # Total energy of system
    df['E_tot_i_memb'] = df['Em_tot_i_memb'] + df['Es_seg_sum_i']
    df['E_tot_i_mm'] = df['Em_tot_i_mm'] + df['Es_seg_sum_i']
    df['E_tot_i_comp'] = df['Em_tot_i_comp'] + df['Es_seg_sum_i']

    return df


def plot_elastic_energy_by_depth(df, path_save, save_id, z_clip=2.5):
    px = 'dZ'
    py1s = ['Em_seg_i_memb', 'Em_seg_i_metal', 'Em_seg_i_mm', 'Em_seg_i_comp']
    py2s = ['Em_seg_sum_i_memb', 'Em_seg_sum_i_metal', 'Em_seg_sum_i_mm', 'Em_seg_sum_i_comp']
    py3s = ['Em_flat_i_memb', 'Em_flat_i_metal', 'Em_flat_i_mm', 'Em_flat_i_comp']
    py4s = ['Em_tot_i_memb', 'Em_tot_i_metal', 'Em_tot_i_mm', 'Em_tot_i_comp']

    df = df[df[px] < df[px].max() - z_clip * 1e-6]
    x = df[px] * 1e6

    pys = [py1s, py2s, py3s, py4s]
    fig, axs = plt.subplots(nrows=len(pys), figsize=(4, len(pys) * 2), sharex=True)

    for ax, pys in zip(axs, pys):
        for py in pys:
            ax.plot(x, df[py], label=py)
        ax.set_ylabel('Energy (J)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize='xx-small')
    axs[-1].set_xlabel('Depth (um)')
    plt.tight_layout()
    plt.savefig(join(path_save, save_id), dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()


def plot_elastic_energy_by_depth_by_component(df, path_save, save_id, z_clip=2.5):
    px = 'dZ'

    py1s = [['Em_seg_i_metal'], ['Em_seg_i_memb', 'Em_seg_i_mm'], ['Em_seg_i_comp']]
    py2s = [['Em_seg_sum_i_metal'], ['Em_seg_sum_i_memb', 'Em_seg_sum_i_mm'], ['Em_seg_sum_i_comp']]
    py3s = [['Em_flat_i_metal'], ['Em_flat_i_memb', 'Em_flat_i_mm'], ['Em_flat_i_comp']]
    py4s = [['Em_tot_i_metal'], ['Em_tot_i_memb', 'Em_tot_i_mm'], ['Em_tot_i_comp']]

    df = df[df[px] < df[px].max() - z_clip * 1e-6]
    x = df[px] * 1e6

    pys = [py1s, py2s, py3s, py4s]
    fig, axes = plt.subplots(nrows=len(pys), ncols=3, figsize=(10, len(pys) * 2), sharex=True)

    for row, pys in zip(axes, pys):
        for ax, py in zip(row, pys):
            ax.plot(x, df[py], label=py)
            ax.grid(alpha=0.25)
            ax.legend(fontsize='x-small')
    for i in range(3):
        axes[i, 0].set_ylabel('Energy (J)')
        axes[-1, i].set_xlabel('Depth (um)')
    plt.tight_layout()
    plt.savefig(join(path_save, save_id), dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()


def plot_electrostatic_energy_by_depth(df, path_save, save_id, z_clip=2.5):
    px = 'dZ'
    py1s = ['Es_seg_i']
    py2s = ['Es_seg_sum_i']

    df = df[df[px] < df[px].max() - z_clip * 1e-6]
    x = df[px] * 1e6

    pys = [py1s, py2s]
    fig, axs = plt.subplots(nrows=len(pys), figsize=(4, len(pys) * 2), sharex=True)

    for ax, pys in zip(axs, pys):
        for py in pys:
            ax.plot(x, df[py], label=py)
        ax.set_ylabel('Energy (J)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize='xx-small')
    axs[-1].set_xlabel('Depth (um)')
    plt.tight_layout()
    plt.savefig(join(path_save, save_id), dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()


def plot_electrostatic_energy_by_depth_overlay_by_voltage(df, voltages, path_save, save_id, z_clip=2.5):
    px = 'dZ'
    pv = 'U'
    pys = ['Es_seg_i', 'Es_seg_sum_i']

    df = df[df[px] < df[px].max() - z_clip * 1e-6]

    fig, axs = plt.subplots(nrows=len(pys), figsize=(4, len(pys) * 2.75), sharex=True)
    for ax, py in zip(axs, pys):
        for voltage in voltages:
            df_v = df[df[pv] == voltage]
            ax.plot(df_v[px] * 1e6, df_v[py], lw=0.75, label=f'{voltage} V')
        ax.set_ylabel('Energy (J)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize='xx-small', title=py, title_fontsize='xx-small')  # loc='upper left', bbox_to_anchor=(1, 1),
    axs[-1].set_xlabel('Depth (um)')
    plt.tight_layout()
    plt.savefig(join(path_save, save_id), dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()


def plot_total_energy_by_depth(df, path_save, save_id, z_clip=2.5):
    px = 'dZ'
    py1s = ['Em_seg_i_memb', 'Em_seg_i_mm', 'Em_seg_i_comp']  #, 'Es_seg_i']
    py2bs = ['Em_seg_sum_i_memb', 'Em_seg_sum_i_mm', 'Em_seg_sum_i_comp']  # , 'Es_seg_sum_i']
    py2s = ['Em_flat_i_memb', 'Em_flat_i_mm', 'Em_flat_i_comp']  # , 'Es_seg_sum_i']
    py3s = ['Em_tot_i_memb', 'Em_tot_i_mm', 'Em_tot_i_comp']  # , 'Es_seg_sum_i']
    py4s = ['E_tot_i_memb', 'E_tot_i_mm', 'E_tot_i_comp']

    df = df[df[px] < df[px].max() - z_clip * 1e-6]
    x = df[px] * 1e6

    pyss = [py1s, py2bs, py2s, py3s, py4s]
    fig, axs = plt.subplots(nrows=len(pyss), figsize=(4, len(pyss) * 2), sharex=True)

    for ax, pys in zip(axs, pyss):
        for py in pys:
            ax.plot(x, df[py], label=py)
        ax.set_ylabel('Energy (J)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize='xx-small')
    axs[-1].set_xlabel('Depth (um)')
    plt.tight_layout()
    plt.savefig(join(path_save, save_id), dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()


def plot_total_energy_by_depth_overlay_by_voltage(df, voltages, path_save, save_id, z_clip=2.5):
    px = 'dZ'
    pv = 'U'
    pys = ['E_tot_i_memb', 'E_tot_i_mm', 'E_tot_i_comp']

    df = df[df[px] < df[px].max() - z_clip * 1e-6]

    fig, axs = plt.subplots(nrows=len(pys), figsize=(4, len(pys) * 2.75), sharex=True)
    for ax, py in zip(axs, pys):
        for voltage in voltages:
            df_v = df[df[pv] == voltage]
            ax.plot(df_v[px] * 1e6, df_v[py], lw=0.75, label=f'{voltage} V')
        ax.set_ylabel('Energy (J)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize='xx-small', title=py, title_fontsize='xx-small')  # loc='upper left', bbox_to_anchor=(1, 1),
    axs[-1].set_xlabel('Depth (um)')
    plt.tight_layout()
    plt.savefig(join(path_save, save_id + '.png'), dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()


def plot_normalized_total_energy_by_depth_by_voltage_overlay_by_model(df, voltages, path_save, save_id, z_clip=2.5):
    px = 'dZ'
    pv = 'U'
    pys = ['E_tot_i_memb', 'E_tot_i_mm', 'E_tot_i_comp']

    df = df[df[px] < df[px].max() - z_clip * 1e-6]

    fig, axs = plt.subplots(nrows=len(voltages), figsize=(4, len(voltages) * 2.75), sharex=True)
    for ax, voltage in zip(axs, voltages):
        df_v = df[df[pv] == voltage]
        for py in pys:
            ax.plot(df_v[px] * 1e6, df_v[py] / df_v[py].max(), lw=0.75, label=py)
        ax.set_ylabel(r'$\tilde{E}$')
        ax.grid(alpha=0.25)
        ax.legend(fontsize='xx-small', title=f'{voltage} V', title_fontsize='xx-small')  # loc='upper left', bbox_to_anchor=(1, 1),
    axs[-1].set_xlabel('Depth (um)')
    plt.tight_layout()
    plt.savefig(join(path_save, save_id + '.png'), dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()


def animate_normalized_total_energy_by_depth_by_voltage_overlay_by_model(df, voltages, path_save, save_id, z_clip=2.5):
    path_save = join(path_save, 'animate_norm')
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    px = 'dZ'
    pv = 'U'
    pys = ['E_tot_i_memb', 'E_tot_i_mm', 'E_tot_i_comp']
    lss = ['k-', 'r-', 'b-']

    df = df[df[px] < df[px].max() - z_clip * 1e-6]

    for voltage in voltages:
        df_v = df[df[pv] == voltage]
        fig, axs = plt.subplots(ncols=3, figsize=(6, 2.75), sharey=True)
        for ax, py, ls in zip(axs, pys, lss):
            ax.plot(df_v[px] * 1e6, df_v[py] / df_v[py].max(), ls)
            ax.grid(alpha=0.25)
            ax.set_title(py, fontsize='x-small')
            ax.set_xlabel('Depth (um)')
        axs[0].set_ylim([0.65, 1.05])
        axs[0].set_yticks([0.7, 0.8, 0.9, 1.0])
        axs[0].set_ylabel(r'$\tilde{E}$', labelpad=-2)
        axs[0].text(-0.05, 1.05, f'{voltage}V', color='black', fontsize=8,
                horizontalalignment='left', verticalalignment='bottom', transform=axs[0].transAxes)
        plt.tight_layout()
        plt.savefig(join(path_save, save_id + f'_{voltage}V.png'), dpi=300, facecolor='w', bbox_inches='tight')
        plt.close()


def plot_normalized_total_energy_by_depth(df, path_save, save_id, z_clip=2.5):
    px = 'dZ'
    py1s = ['Em_seg_i_memb', 'Em_seg_i_mm', 'Em_seg_i_comp']  #, 'Es_seg_i']
    py2bs = ['Em_seg_sum_i_memb', 'Em_seg_sum_i_mm', 'Em_seg_sum_i_comp']  # , 'Es_seg_sum_i']
    py2s = ['Em_flat_i_memb', 'Em_flat_i_mm', 'Em_flat_i_comp']  # , 'Es_seg_sum_i']
    py3s = ['Em_tot_i_memb', 'Em_tot_i_mm', 'Em_tot_i_comp']  # , 'Es_seg_sum_i']
    py4s = ['E_tot_i_memb', 'E_tot_i_mm', 'E_tot_i_comp']
    pyb = 'Es_seg_sum_i'

    df = df[df[px] < df[px].max() - z_clip * 1e-6]
    x = df[px] * 1e6

    pyss = [py1s, py2bs, py2s, py3s, py4s]
    ii = np.arange(len(pyss))
    fig, axs = plt.subplots(nrows=len(pyss), figsize=(4.5, len(pyss) * 2), sharex=True)

    for i, ax, pys in zip(ii, axs, pyss):
        for py in pys:
            ax.plot(x, df[py] / df[py].max(), label=py)
        if i > 2:
            ax.plot(x, df[pyb] / df[pyb].min(), label=pyb)
        ax.set_ylabel(r'$\tilde{E}$')
        ax.grid(alpha=0.25)
        ax.legend(fontsize='xx-small')
    axs[-1].set_xlabel('Depth (um)')
    plt.tight_layout()
    plt.savefig(join(path_save, save_id), dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()


def plot_total_energy_by_depth_by_component(df, path_save, save_id, z_clip=2.5):
    px = 'dZ'
    py1s = ['Em_tot_i_memb', 'Em_tot_i_mm', 'Em_tot_i_comp']
    py2s = ['E_tot_i_memb', 'E_tot_i_mm', 'E_tot_i_comp']
    pyb = 'Es_seg_sum_i'

    df = df[df[px] < df[px].max() - z_clip * 1e-6]
    x = df[px] * 1e6

    pyss = [py1s, py2s]
    fig, axes = plt.subplots(nrows=len(pyss), ncols=3, figsize=(10, len(pyss) * 2.5), sharex=True)

    for row, pys in zip(axes, pyss):
        for ax, py in zip(row, pys):
            ax.plot(x, df[py], label=py)
            ax.grid(alpha=0.25)
            ax.legend(fontsize='xx-small')
    for i in range(3):
        axr = axes[0, i].twinx()
        axr.plot(x, df[pyb], color='m', linestyle='-', lw=0.75, alpha=0.5)
        axes[-1, i].set_xlabel('Depth (um)')
    for j in range(2):
        axes[j, 0].set_ylabel('Energy (J)')
    plt.tight_layout()
    plt.savefig(join(path_save, save_id), dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()


def wrapper_calculate_total_energy_by_depth_by_voltage(df_e, df_vol, dict_electrostatic, voltages):
    outputs_electrostatic = ['Es_seg_i', 'Es_seg_sum_i']
    df_te_vs = []
    for voltage in voltages:
        # Electrostatic energy
        df_es_v = calculate_electrostatic_energy_by_depth_for_voltage(df_vol, dict_electrostatic, voltage)
        # Total energy
        df_te_v = calculate_total_energy_by_depth(df=df_e.join(df_es_v[outputs_electrostatic]))
        # Append to list
        df_te_v['U'] = voltage
        df_te_vs.append(df_te_v)

    df_te_vs = pd.concat(df_te_vs)
    df_te_vs = df_te_vs.reset_index(drop=True)

    return df_te_vs


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


def wrapper_find_first_energy_minima_by_voltage(df, ignore_dZ_below_v, assign_z='z_comp'):

    px = 'dZ'
    pv = 'U'
    pys = ['E_tot_i_memb', 'E_tot_i_mm', 'E_tot_i_comp']
    pxys = ['z_memb', 'z_mm', 'z_comp']

    voltages = df[pv].unique()
    dZ_pys = []
    for py in pys:
        dZ_is = []
        dZ_i = 0  # moving deflection
        for voltage in voltages:
            df_v = df[df[pv] == voltage]

            # Find the first energy minima
            first_minima_x, first_minima_y = find_first_energy_minima(
                x=df_v[px].to_numpy(),
                y=df_v[py].to_numpy(),
                print_id=f'({voltage}V)',
            )

            if not np.isnan(first_minima_x):
                if (
                        dZ_i == 0 and                                               # if previous position was z=0
                        voltage < ignore_dZ_below_v[1] and                 # if voltage is "low"
                        first_minima_x > df[px].max() - ignore_dZ_below_v[0]   # if first minima occurs near pull-in
                ):
                    dZ_i = 0  # ignore the minima
                else:
                    dZ_i = first_minima_x
            elif np.isnan(first_minima_x) and dZ_i != 0:
                dZ_i = df_v[px].max()
            else:
                pass  # dZ_i remains at 0.

            dZ_is.append(dZ_i)

        dZ_pys.append(dZ_is)

    df_dZ = pd.DataFrame(np.vstack([voltages, dZ_pys[0], dZ_pys[1], dZ_pys[2]]).T,
                         columns=[pv] + pxys)

    df_dZ['z'] = df_dZ[assign_z]

    return df_dZ


def plot_total_energy_and_minima_by_depth(voltages, df_te_vs, df_dz_vs, path_save, save_id, z_clip=2.5):
    pv = 'U'
    px = 'dZ'
    pys = ['E_tot_i_memb', 'E_tot_i_mm', 'E_tot_i_comp']
    pzs = ['z_memb', 'z_mm', 'z_comp']
    df_te_vs = df_te_vs[df_te_vs[px] < df_te_vs[px].max() - z_clip * 1e-6]

    for voltage in voltages:
        df = df_te_vs[df_te_vs[pv] == voltage].reset_index(drop=True)
        x = df[px] * 1e6
        fig, axs = plt.subplots(nrows=len(pys), figsize=(4, len(pys) * 2), sharex=True)
        for ax, py, pz in zip(axs, pys, pzs):
            # z of first minima
            y_dZ = df_dz_vs.loc[df_dz_vs[pv] == voltage, pz].values[0]
            # total energy of first minima
            te_dZ = df[py].iloc[(df[px] - y_dZ).abs().idxmin()]
            ax.plot(x, df[py], label=py)
            ax.plot(y_dZ * 1e6, te_dZ, 'r*', label=f'z: {np.round(y_dZ * 1e6, 1)} ' + r' $\mu$m')
            ax.set_ylabel('Energy (J)')
            ax.grid(alpha=0.25)
            ax.legend(fontsize='xx-small')
        axs[-1].set_xlabel('Depth (um)')
        plt.tight_layout()
        plt.savefig(join(path_save, save_id + f'_{voltage}V.png'), dpi=300, facecolor='w', bbox_inches='tight')
        plt.close()


def animate_total_energy_and_minima_by_depth(voltages, df_te_vs, df_dz_vs, path_save, save_id, z_clip=2.5):
    path_save = join(path_save, 'animate_minima')
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    pv = 'U'
    px = 'dZ'
    pys = ['E_tot_i_memb', 'E_tot_i_mm', 'E_tot_i_comp']
    pzs = ['z_memb', 'z_mm', 'z_comp']
    lss = ['g-', 'r-', 'b-']
    df_te_vs = df_te_vs[df_te_vs[px] < df_te_vs[px].max() - z_clip * 1e-6]

    y_limits = []
    for py, pz in zip(pys, pzs):
        index_pull_in = df_dz_vs[df_dz_vs[pz] > df_dz_vs[pz].max() - z_clip * 2e-6].index.min()
        voltage_pull_in = df_dz_vs.loc[index_pull_in, pv]
        energy_min_pull_in = df_te_vs[df_te_vs[pv] <= voltage_pull_in][py].min()
        energy_max = df_te_vs[df_te_vs[pv] <= voltage_pull_in][py].max()
        energy_span = energy_max - energy_min_pull_in
        y_limits.append([energy_min_pull_in - energy_span / 5, energy_max + energy_span / 20])

    for voltage in voltages:
        df = df_te_vs[df_te_vs[pv] == voltage].reset_index(drop=True)
        x = df[px] * 1e6
        fig, axs = plt.subplots(nrows=len(pys), figsize=(4, len(pys) * 2), sharex=True)
        for ax, py, pz, y_lim, ls in zip(axs, pys, pzs, y_limits, lss):
            # z of first minima
            y_dZ = df_dz_vs.loc[df_dz_vs[pv] == voltage, pz].values[0]
            # total energy of first minima
            te_dZ = df[py].iloc[(df[px] - y_dZ).abs().idxmin()]
            ax.plot(x, df[py], ls, label=py)
            ax.plot(y_dZ * 1e6, te_dZ, 'k*', label=f'z: {np.round(y_dZ * 1e6, 1)} ' + r' $\mu$m')
            ax.set_ylabel('Energy (J)')
            ax.set_ylim(y_lim)
            ax.grid(alpha=0.25)
            ax.legend(loc='upper left', fontsize='xx-small')
        axs[-1].set_xlabel('Depth (um)')
        plt.tight_layout()
        plt.savefig(join(path_save, save_id + f'_{voltage}V.png'),
                    dpi=300, facecolor='w', bbox_inches='tight')
        plt.close()


def plot_z_by_voltage_by_model(df, save_dir, save_id):
    pv = 'U'
    pys = ['z_memb', 'z_mm', 'z_comp']
    fig, ax = plt.subplots(figsize=(4, 3))
    for py in pys:
        ax.plot(df[pv], df[py] * 1e6 * -1, label=py)
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel(r'z ($\mu$m)')
    ax.grid(alpha=0.25)
    ax.legend(fontsize='x-small')
    plt.tight_layout()
    plt.savefig(join(save_dir, save_id + '_model_z-by-v-by-model.png'), dpi=300, facecolor='w')


# ------------------- COMPARE (PLOT) CONFIGURATIONS

def plot_sweep_z_by_voltage_by_model(df, key, labels, units, save_dir, save_id):
    pv = 'U'
    pys = ['z_memb', 'z_mm', 'z_comp']
    svs = df[key].unique()

    if key == 'memb_E':
        lgnd_idx = 0
    elif key in ['met_E']:
        lgnd_idx = 1
    elif key == 'comp_E':
        lgnd_idx = 2
    else:
        lgnd_idx = 0
    if len(units) > 1:
        lgnd_title = f'{key} ({units})'
    else:
        lgnd_title = key

    fig, axs = plt.subplots(nrows=len(pys), sharex=True, figsize=(4, 6.5))

    for ax, py in zip(axs, pys):
        for sv, lbl in zip(svs, labels):
            df_sv = df[df[sweep_key] == sv]
            ax.plot(df_sv[pv], df_sv[py] * 1e6 * -1, label=lbl)
        ax.text(0.95, 0.95, py, color='black', fontsize=8,
                 horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.grid(alpha=0.25)

    axs[lgnd_idx].legend(fontsize='x-small', title=lgnd_title, title_fontsize='x-small')
    axs[-1].set_xlabel('Voltage (V)')
    plt.tight_layout()
    plt.savefig(join(save_dir, save_id + f'_model_z-by-v-by-model_sweep-{sweep_key}.png'),
                dpi=300, facecolor='w', bbox_inches='tight')

def plot_sweep_z_by_voltage_comp_E(df, key, labels, units, save_dir, save_id):
    pv = 'U'
    py = 'z_comp'
    svs = df[key].unique()

    if len(units) > 1:
        lgnd_title = f'{key} ({units})'
    else:
        lgnd_title = key

    fig, ax = plt.subplots(figsize=(3.3, 2.625))
    for sv, lbl in zip(svs, labels):
        df_sv = df[df[sweep_key] == sv]
        ax.plot(df_sv[pv], df_sv[py] * 1e6 * -1, label=lbl)
    ax.set_ylabel(r'$z \: (\mu m)$')
    ax.legend(fontsize='x-small', title=lgnd_title, title_fontsize='x-small', loc='upper right', frameon=False)
    ax.set_xlabel('Voltage (V)')
    plt.tight_layout()
    plt.savefig(join(save_dir, save_id + f'_z-by-v-by-compE_sweep-{sweep_key}.png'),
                dpi=300, facecolor='w', bbox_inches='tight')
