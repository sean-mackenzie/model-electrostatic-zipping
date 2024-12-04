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