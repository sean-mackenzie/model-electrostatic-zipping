# tests/test_model_sweep.py

# imports
import numpy as np
from scipy.special import erf


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


def annular_area(radius_m: float, width_m: float) -> float:
    """
    Calculates the area of a circular annular region (ring-shaped),
    where the annulus is centered at 'radius_m' and has total width 'width_m'.

    Parameters:
    -----------
    radius_m : float
        Central radius of the annular region, in meters.
    width_m : float
        Total width of the annular ring, in meters.

    Returns:
    --------
    area_m2 : float
        Area of the annular region, in square meters (m^2).
    """
    r_outer = radius_m + width_m / 2
    r_inner = radius_m - width_m / 2
    area_m2 = np.pi * (r_outer**2 - r_inner**2)
    return area_m2


def radius_of_curvature_and_curvature_from_angle_and_arc_length(angle_degrees, arc_length):
    """
    Calculates the radius of curvature and curvature for a circular arc given the
    angle in degrees and the arc length. The calculations assume the arc geometry
    is part of a perfect circle.

    Example: Assume our 20-um thick membrane bends to conform against a sidewall
    having a slope angle of 10 degrees. Let's assume that this occurs over a
    length of 2X the membrane thickness on either side of the bend, equating
    to a total length of 80 um. In this scenario, the input 'angle_degrees' is
    10 degrees, and the input 'arc_length' is 80 um.

    :param angle_degrees: The angle in degrees subtended by the arc at the center
    :param arc_length: The length of the arc
    :return: A tuple containing the radius of curvature and curvature, in that
             order
    """
    angle_radians = np.deg2rad(angle_degrees)  # = angle_degrees * np.pi / 180
    # for a circular arc, the arc length (s) = R * theta
    radius_of_curvature = arc_length / angle_radians
    curvature = 1 / radius_of_curvature
    return radius_of_curvature, curvature


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


def complete_erf_profile(px, py, width, dia_flat):
    """ px_revolved, py_revolved = complete_erf_profile(px, py, width, dia_flat) """
    px_revolved = np.flip(px) + width / 2 + dia_flat / 2
    py_revolved = py

    px_revolved = np.append(px_revolved, px[-1])
    py_revolved = np.append(py_revolved, py[-1])

    return px_revolved, py_revolved