import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import BSpline


def read_tck(filepath):
    df = pd.read_excel(filepath)
    t = df.t.to_numpy()
    c = df.c.to_numpy()
    k = int(filepath[-6])
    tck = (t, c, k)
    return tck


def profile_tck(tck, x):
    return BSpline(*tck)(x)


def profile_tck2solver(r, z):
    r = (r - np.max(r)) * -1
    z = z - np.max(z)

    r = np.flip(r)
    z = np.flip(z)

    r = r[:np.argmin(z)]
    z = z[:np.argmin(z)]

    return r, z


def dict_from_tck(wid, fid, depth, radius, units, num_segments, fp_tck, r_min=None):
    tck = read_tck(filepath=fp_tck)

    if r_min is None:
        r_min = 0
    r = np.linspace(r_min, radius, num_segments)
    z = profile_tck(tck, r)
    rnew, znew = profile_tck2solver(r, z)

    rnew = rnew * units
    znew = znew * units

    dict_fid = {
        'wid': wid,
        'fid': fid,
        'depth': depth,
        'radius': radius,
        'units': units,
        'tck': tck,
        'r_raw': r,
        'z_raw': z,
        'r': rnew,
        'z': znew,
    }

    return dict_fid


# --- SURFACE PROFILES

def read_surface_profile(dict_settings, subset=None, hole=True, fid_override=None):
    # read surface profile
    df = pd.read_excel(dict_settings['path_process_profiles'])
    # -
    # get feature profile
    if fid_override is not None:
        fid = fid_override
    else:
        fid = dict_settings['fid']
    if 'step_process_profile' in dict_settings.keys():
        step = dict_settings['step_process_profile']
    else:
        step = df['step'].max()
    df = df[(df['fid'] == fid) & (df['step'] == step)]
    # -
    # keep only necessary columns
    df = df[['z', 'r']]

    # filter out profile where through-hole has been etched
    if hole:
        df = df[df['r'].abs() > dict_settings['radius_hole_microns']]

    # return a subset of the profile
    if subset is None:
        pass  # return the full-width profile
    elif subset == 'full':
        pass  # alternate method to return the full-width profile
    elif subset == 'abs':
        df['r'] = df['r'].abs()
    elif subset == 'right_half':
        df = df[df['r'] > 0]
    elif subset == 'left_half':
        df = df[df['r'] < 0]
        df['r'] = df['r'].abs()
    elif subset == 'average_halves':
        smooth_r, smooth_z = smooth_symmetric_profile_with_tolerance(df['r'].to_numpy(), df['z'].to_numpy(), tolerance=2)
        df = pd.DataFrame({'r': smooth_r, 'z': smooth_z})
        df = df[df['r'] > 0]
    else:
        raise ValueError("Options are: [None or full, abs, right_half, left_half]")

    return df

def smooth_symmetric_profile_with_tolerance(r, z, tolerance=2):
        """
        Smooths a profile by mirroring it about r=0 and averaging within a tolerance range.

        Parameters:
        r (array-like)       : Array of r values (e.g., -1000 to 1000, non-symmetric).
        z (array-like)       : Corresponding array of z values.
        tolerance (float)    : Range of r values (±tolerance) to average for smoothing.

        Returns:
        r_smooth (numpy.ndarray): Array of smoothed r values (from input data).
        z_smooth (numpy.ndarray): Array of averaged z values corresponding to r_smooth.
        """
        # Ensure inputs are numpy arrays
        #r = np.array(r)
        #z = np.array(z)

        # Create the mirrored profile
        r_mirrored = -r
        z_mirrored = z  # z values remain the same

        # Combine original and mirrored profiles
        r_combined = np.concatenate((r, r_mirrored))
        z_combined = np.concatenate((z, z_mirrored))

        # Sort by r
        sorted_indices = np.argsort(r_combined)
        r_sorted = r_combined[sorted_indices]
        z_sorted = z_combined[sorted_indices]

        # Initialize smoothed profile arrays
        r_smooth = []
        z_smooth = []

        # Perform averaging within the tolerance
        for r_value in r:
            # Find indices where r_combined is within tolerance of r_value
            mask = np.abs(r_sorted - r_value) <= tolerance

            # If any points fall within the tolerance range, compute the mean z
            if np.any(mask):
                z_mean = np.mean(z_sorted[mask])
                r_smooth.append(r_value)
                z_smooth.append(z_mean)

        return np.array(r_smooth), np.array(z_smooth)


def get_surface_profile_dict(dict_settings, subset='right', include_hole=True, shift_r=0, shift_z=0, scale_z=1):
    if 'fid_process_profile' in dict_settings.keys():
        surf_fid_override = dict_settings['fid_process_profile']
    else:
        surf_fid_override = None
    if subset == 'full':
        include_hole = False

    df_surface = read_surface_profile(
        dict_settings,
        subset=subset,
        hole=include_hole,
        fid_override=surf_fid_override,
    )
    dict_surface_profilometry = {'r': df_surface['r'].to_numpy(), 'z': df_surface['z'].to_numpy(),
                                 'dr': shift_r, 'dz': shift_z, 'scale_z': scale_z,
                                 'subset': subset}
    return dict_surface_profilometry

def get_zipping_interface_r_from_z(z0, surf_r, surf_z, z0_max=-0.01):
    if z0 > z0_max:
        z0 = z0_max
    # make sure the arrays are sorted
    # primary sort: r
    sorted_indices = np.argsort(surf_r)
    # Use these indices to sort both arrays
    sorted_r = surf_r[sorted_indices]
    sorted_z = surf_z[sorted_indices]
    # find smallest r that satisfies
    rmin_nearest = sorted_r[np.argmax(np.abs(sorted_z - z0) < 2)]
    return rmin_nearest


def get_zipping_interface_rz(r, z, surf_r, surf_z):
    # If you get problems, try using the code below to enforce limits on r, z
    """
    # make sure the arrays are sorted
    surf_z = surf_z[np.argsort(surf_r)]
    surf_r = surf_r[np.argsort(surf_r)]
    # define the maximum allowable zipping interface
    surf_r_max = surf_r[np.argmax(surf_z < -2)]
    surf_z_max = surf_z[np.argmax(surf_z < -2)]
    """

    # 0th-pass: estimate r, z as average of particles nearest the center
    """ NOTE: if np.percentile() gives an IndexError, the this frame may have no data.
    This happened for 02242025_W13-D1-C17-20pT: TID24 Frame 54"""
    zipping_interface_r = np.percentile(r, 7.5)
    zipping_interface_z = np.mean(z[r < zipping_interface_r])

    # 1st-pass: get zipping interface
    zipping_interface_r = get_zipping_interface_r_from_z(
        z0=zipping_interface_z,
        surf_r=surf_r,
        surf_z=surf_z,
    )
    try:
        # 2nd-pass: refine zipping interface
        r_fringing = np.percentile(r[r < zipping_interface_r], 75)
        # r_midpoint = (zipping_interface_r + np.min(r)) / 2
        zipping_interface_z = np.mean(z[(r < zipping_interface_r) & (r > r_fringing)])
        zipping_interface_r = get_zipping_interface_r_from_z(
            z0=zipping_interface_z,
            surf_r=surf_r,
            surf_z=surf_z,
        )
    except IndexError:
        pass
    return zipping_interface_r, zipping_interface_z


def get_apparent_radial_displacement_due_to_rotation_function(surf_r, surf_z, poly_deg, membrane_thickness, z_clip=-0.25):
    # Clip surface profile data to only include the sloped portion
    x = surf_r[surf_z < z_clip]
    y = surf_z[surf_z < z_clip]

    # Fit data to a polynomial
    coefficients = np.polyfit(x, y, deg=poly_deg)
    polynomial = np.poly1d(coefficients)

    # Derivative of the fitted polynomial
    polynomial_derivative = polynomial.deriv()

    # I'm not sure if this function setup ever worked.
    """
    def apparent_radial_displacement(r):
        r_bounds = (np.min(x), np.max(x))
        if r < r_bounds[0] or r > r_bounds[1]:
            return 0
        else:
            return membrane_thickness * np.sin(np.arctan(polynomial_derivative(r)))
    """

    # Define a function that gives the apparent radial displacement
    def apparent_radial_displacement(r):
        rmin = x.min()
        rmax = x.max()
        # Define a helper function
        def process_element(xn, lower_limit, upper_limit):
            if lower_limit <= xn <= upper_limit:
                return membrane_thickness * np.sin(np.arctan(polynomial_derivative(xn)))
            return 0
        return r.apply(lambda xm: process_element(xm, lower_limit=rmin, upper_limit=rmax))

    return apparent_radial_displacement


def calculate_apparent_radial_displacement_due_to_rotation(surf_r, surf_z, poly_deg, membrane_thickness, z_clip=-0.25,
                                                           path_save=None):
    # Clip surface profile data to only include the sloped portion
    x = surf_r[surf_z < z_clip]
    y = surf_z[surf_z < z_clip]

    # Fit data to a polynomial
    coefficients = np.polyfit(x, y, deg=poly_deg)
    polynomial = np.poly1d(coefficients)

    # Derivative of the fitted polynomial
    polynomial_derivative = polynomial.deriv()

    # Define a function that gives the apparent radial displacement
    def apparent_radial_displacement(r):
        rmin = x.min()
        rmax = x.max()
        if poly_deg == 3: rmax += 5
        # Define a helper function
        def process_element(xn, lower_limit, upper_limit):
            if lower_limit <= xn <= upper_limit:
                return membrane_thickness * np.sin(np.arctan(polynomial_derivative(xn)))
            return 0
        return r.apply(lambda xm: process_element(xm, lower_limit=rmin, upper_limit=rmax))

    # Angle calculated from slope using arctan
    angle_radians = np.arctan(polynomial_derivative(x))  # Angle in radians
    angle_degrees = np.degrees(angle_radians)  # Convert to degrees

    # Apparent radial displacement calculated from membrane thickness + slope angle
    apparent_displacement = membrane_thickness * np.sin(angle_radians)

    # Plotting original data, fitted function, and derivative
    import matplotlib.pyplot as plt
    from os.path import join
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(4.5, 6))  # (7, 8)
    # ax1.scatter(x, y, s=4, alpha=0.95, label='Surface profile')
    ax1.plot(x, y, 'k-', label='Surface profile')
    ax1.plot(x, polynomial(x), 'r-', lw=0.85, label=f'{poly_deg}-deg Polynominal')
    ax2.plot(x, polynomial_derivative(x), label='Derivative (Slope)', color='green', linestyle='--')
    ax3.plot(x, angle_degrees, label='Angle (degrees)', color='purple')
    ax4.plot(x, apparent_displacement, label='Apparent Displacement', color='orange')
    ax4.axvline(x.min(), label='rmin={}'.format(np.round(x.min(), 1)), color='k', lw=0.5, linestyle='--')
    ax4.axvline(x.max(), label='rmax={}'.format(np.round(x.max(), 1)), color='k', lw=0.5, linestyle='--')
    ax1.legend(fontsize='x-small')
    ax2.legend(fontsize='x-small')
    ax4.legend(fontsize='xx-small', loc='upper left')
    ax1.set_ylabel(r'$z \: (\mu m)$')
    ax2.set_ylabel('slope')
    ax3.set_ylabel('angle (degrees)')
    ax4.set_ylabel(r'$\Delta_{rot} r \: (\mu m)$')
    ax4.set_xlabel(r'$r \: (\mu m)$')
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, f'apparent_radial_displacement_deg={poly_deg}.png'),
                    dpi=300, facecolor='w', bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    return apparent_radial_displacement

