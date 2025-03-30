import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ----- Physical Parameters -----
R = 1e-3  # Membrane radius [m]
g0 = 100e-6  # Central gap [m] = basin depth
eps0 = 8.854e-12  # Vacuum permittivity [F/m]
Er = 1.0  # Relative permittivity
E_eff = 2e6  # Effective Young's modulus [Pa]
h = 20e-6  # Total membrane thickness [m]
D = E_eff * h ** 3 / (12 * (1 - 0.49 ** 2))  # Effective bending stiffness [Nm]
T = 5  # Pre-stretch tension [N/m]
theta_deg = 10  # Basin sidewall angle
kc = 1e10  # Contact penalty stiffness

# ----- Numerical Grid -----
N = 300
r = np.linspace(1e-6, R, N)
dr = r[1] - r[0]

# ----- Basin Geometry -----
theta_rad = np.radians(theta_deg)
z_basin = g0 - r * np.tan(theta_rad)
z_basin = np.maximum(z_basin, 0)


# ----- Energy Functional with Contact -----
def make_total_energy(V):
    def total_energy(w):
        dw = np.gradient(w, dr)
        d2w = np.gradient(dw, dr)

        # Strain energy
        U_strain = np.sum(0.5 * E_eff * h * (dw ** 4) * r * dr)

        # Pre-stretch tension energy
        U_tension = np.sum(0.5 * T * (dw ** 2) * r * dr)

        # Bending energy
        laplacian = d2w + (1 / r) * dw
        U_bend = np.sum(D * laplacian ** 2 * r * dr)

        # Electrostatic energy
        gap = z_basin - w
        gap = np.maximum(gap, 1e-9)  # Avoid division by zero
        U_elec = -0.5 * eps0 * Er * V ** 2 * np.sum((1.0 / gap) * r * dr)

        # Contact penalty (only when w > basin floor)
        delta = np.maximum(0, w - z_basin)
        U_contact = 0.5 * kc * np.sum(delta ** 2 * r * dr)

        return U_strain + U_tension + U_bend + U_elec + U_contact

    return total_energy


# ----- Voltage Sweep -----
voltages = np.linspace(0, 300, 31)  # From 0 to 300 V in 10 V steps
w_central = []
w_profiles = []
success_flags = []

w_guess = np.zeros_like(r)

for V in voltages:
    print(f"Solving for V = {V:.1f} V...")
    energy_fn = make_total_energy(V)
    res = minimize(energy_fn, w_guess, method='L-BFGS-B', options={'maxiter': 500})

    if res.success:
        w_opt = res.x
        w_guess = w_opt  # Warm start for next iteration
        success_flags.append(True)
    else:
        print(f"  ⚠️ Optimization failed at V = {V:.1f} V.")
        w_opt = w_guess  # Use last good guess
        success_flags.append(False)

    w_profiles.append(w_opt)
    w_central.append(w_opt[0])

# ----- Plot Central Deflection vs Voltage -----
plt.figure(figsize=(6, 4))
plt.plot(voltages, np.array(w_central) * 1e6, marker='o')
plt.xlabel('Voltage [V]')
plt.ylabel('Central Deflection [µm]')
plt.title('Central Membrane Deflection vs Voltage')
plt.grid(True)
plt.tight_layout()
plt.show()
