# plot_arbitrary_data.py

import os
from os.path import join
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

if __name__ == "__main__":
    SAVE_ID = 'C2-0pT'
    PATH_SAVE = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/zipper_paper/Methods/Keithley 2410 4-Wire'
    MATS = ['5/15nm Ti/Au', '5/25nm Ti/Au', '5/35nm Ti/Au']
    THICKNESSES = np.array([15, 25, 35])
    RES = np.array([5.4852, 3.4464, 1.4185])

    def fit_line(x, a, b):
        return a * x + b

    popt, pcov = curve_fit(fit_line, THICKNESSES, RES)

    fig, ax = plt.subplots(figsize=(3.5 * 1.25, 2.75 * 1.25))

    # ax.plot(THICKNESSES, fit_line(THICKNESSES, *popt), 'k-', label='Fit: {}x + {}'.format(round(popt[0], 2), round(popt[1], 2)))
    # ax.plot(THICKNESSES, RES, 'ro', label=SAVE_ID)
    ax.scatter(THICKNESSES[0] + 5, RES[0], label='C2-0pT 5/15nm Ti/Au')
    ax.scatter(THICKNESSES[1] + 5, RES[1], label='C2-0pT 5/25nm Ti/Au')
    ax.scatter(THICKNESSES[2] + 5, RES[2], label='C2-0pT 5/35nm Ti/Au')
    ax.scatter(45 + 5, 20, label='C1-20pT 5/45nm Ti/Au')

    ax.scatter(20, 69, marker='^', label='C7-20pT: MPTMS + 20nm Au')
    ax.scatter(20, 102, marker='^', label='C9-0pT: MPTMS + 20nm Au')

    ax.scatter(25, 215, marker='D', label='C10-20pT: MPTMS + 25nm Au')

    ax.scatter(25, 180, marker='s', label='C13-20pT: MPTMS + 25nm Au')
    ax.scatter(20, 173, marker='s', label='C14-15pT: MPTMS + 20nm Au')
    ax.scatter(25, 162, marker='s', label='C15-15pT: MPTMS + 25nm Au')


    ax.scatter(25, 30, marker='x', label='C17-20pT: MPTMS + 25nm Au')
    ax.scatter(30, 290, marker='x', label='C21-15pT: MPTMS + 30nm Au')

    ax.set_xlabel('Electrode Thickness (nm)')
    ax.set_ylabel('Resistance (Ohms)')
    ax.legend(fontsize='x-small', title='Grouped by Date')
    ax.grid(alpha=0.25)
    # ax.set_title('4-wire Resistance vs. Thickness')
    plt.tight_layout()
    plt.savefig(join(PATH_SAVE, '_Compare-membrane-resistance_all-sub1kOhm.png'), dpi=300, facecolor='w', bbox_inches='tight')
    plt.show()
