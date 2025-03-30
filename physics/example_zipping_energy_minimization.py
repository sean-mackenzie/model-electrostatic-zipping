import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# ----- Parameters -----
R = 1e-3              # Membrane radius [m]
g0 = 100e-6           # Gap at center [m]
E = 1e5               # Young's modulus [Pa]
nu = 0.49             # Poisson's ratio
mu = E / (2 * (1 + nu))  # Shear modulus
Jm = 50               # Gent model parameter
h = 10e-6             # Membrane thickness [m]
eps0 = 8.854e-12      # Vacuum permittivity [F/m]
Er = 1.0              # Relative permittivity

# ----- Discretization -----
N = 300
r = np.linspace(1e-6, R, N)  # avoid r=0
dr = r[1] - r[0]

# ----- Basin Floor (Flat) -----
z_basin = g0 - r * np.tan(np.radians(7))
z_basin = np.maximum(z_basin, 0)

# ----- Assumed Shape Function -----
def shape_profile(w0):
    return w0 * (1 - (r / R)**2)

# ----- Energy Computation -----
def compute_energies(w, V):
    dw = np.gradient(w, dr)
    lam = np.sqrt(1 + dw**2)
    I1 = 2 * lam**2 + lam**(-4)
    if np.any((I1 - 3) >= Jm):
        return np.nan, np.nan
    W = -0.5 * mu * Jm * np.log(1 - (I1 - 3) / Jm)  # J/m³
    U_strain = np.sum(h * W * r * dr)               # J/m

    gap = z_basin - w
    if np.any(gap <= 0):
        return np.nan, np.nan
    U_elec = -0.5 * eps0 * Er * V**2 * np.sum((1.0 / gap) * r * dr)  # J/m

    return U_strain, U_elec

# ----- Solve for V that balances energies for each w0 -----
deflection_range = np.linspace(0, 100e-6, 50)
voltages_required = []
strain_list = []
elec_list = []

for w0 in deflection_range:
    w = shape_profile(w0)

    def objective(V):
        U_s, U_e = compute_energies(w, V)
        if np.isnan(U_s) or np.isnan(U_e):
            return 1e9
        return U_s + U_e

    try:
        sol = root_scalar(objective, bracket=[1, 200], method='brentq')
        V_eq = sol.root
        U_s, U_e = compute_energies(w, V_eq)
        voltages_required.append(V_eq)
        strain_list.append(U_s)
        elec_list.append(U_e)
    except ValueError:
        voltages_required.append(np.nan)
        strain_list.append(np.nan)
        elec_list.append(np.nan)

# ----- Plot Results -----
plt.figure(figsize=(6, 4))
plt.plot(deflection_range * 1e6, voltages_required, marker='o')
plt.xlabel('Assumed Central Deflection [µm]')
plt.ylabel('Required Voltage [V]')
plt.title('Voltage Required for Equilibrium vs Deflection')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6.5, 4.5))
plt.plot(deflection_range * 1e6, strain_list, label='Strain Energy')
plt.plot(deflection_range * 1e6, elec_list, label='Electrostatic Energy')
plt.xlabel('Assumed Central Deflection [µm]')
plt.ylabel('Energy [J/m]')
plt.title('Energy Balance at Equilibrium')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
