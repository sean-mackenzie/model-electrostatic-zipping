import numpy as np
from scipy.interpolate import splrep, splev

def fit_smoothing_spline(x, y, s, xnew):
    """

    :param x:
    :param y:
    :param s: smoothing factor, where higher values equal more smoothing.
    :return:
    """
    tck = splrep(x, y, s=s)
    ynew = splev(xnew, tck, der=0)
    return ynew

def wrapper_fit_radial_membrane_profile(x, y, s, faux_r_zero=None, faux_r_edge=None):
    """

    :param x:
    :param y:
    :param s:
    :param faux_r_zero: y_faux = np.mean(y[x < faux_r_zero]) (e.g., faux_r_zero = dict_settings['radius_hole_microns'])
    :param faux_r_edge: x_faux = faux_r_edge (e.g., faux_r_edge = dict_settings['radius_microns'])
    :return:
    """
    if faux_r_zero is not None:
        x_fake, y_fake = 0, np.mean(y[x < faux_r_zero])
        x = np.append(x_fake, x)
        y = np.append(y_fake, y)
    if faux_r_edge is not None:
        x_fake, y_fake = faux_r_edge, 0
        x = np.append(x, x_fake)
        y = np.append(y, y_fake)

    xnew = np.linspace(x.min(), x.max(), 200)
    ynew = fit_smoothing_spline(x, y, s, xnew)

    return xnew, ynew, x, y

def wrapper_fit_radial_membrane_profile_(x, y, s, dict_settings, faux_r_zero, faux_r_edge):
    if faux_r_zero:
        x_fake, y_fake = 0, np.mean(y[x < dict_settings['radius_hole_microns']])
        x = np.append(x_fake, x)
        y = np.append(y_fake, y)
    if faux_r_edge:
        x_fake, y_fake = dict_settings['radius_microns'], 0
        x = np.append(x, x_fake)
        y = np.append(y, y_fake)

    xnew = np.linspace(x.min(), x.max(), 200)
    ynew = fit_smoothing_spline(x, y, s, xnew)

    return xnew, ynew, x, y