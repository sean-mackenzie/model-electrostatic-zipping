# plot_arbitrary_data.py

import os
from os.path import join
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

if __name__ == "__main__":
    SAVE_ID = 'C2-0pT'
    PATH_SAVE = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/zipper_paper'
    MATS = ['5/15nm Ti/Au', '5/25nm Ti/Au', '5/35nm Ti/Au']
    THICKNESSES = np.linspace(0, 40)

    def func_Osmani_2016(x, a, b, c):
        return a + b * x + c * x**2

    E = func_Osmani_2016(THICKNESSES, 1.1, 0.09, 0.01)
    E_lower = func_Osmani_2016(THICKNESSES, 1.2, 0.11, 0.011)
    E_upper = func_Osmani_2016(THICKNESSES, 1.0, 0.07, 0.009)

    fig, ax = plt.subplots(figsize=(4.5, 3.75))

    ax.plot(THICKNESSES, E, '-', color='tab:red', lw=1.25, label='ELASTOSIL: ' + r'$1.1\pm 0.1$ MPa')
    ax.fill_between(THICKNESSES, E_lower, E_upper, color='red', ec='none', alpha=0.2)

    ax.set_xlabel('Au Thickness (nm)')
    ax.set_ylabel('Elastic modulus E (MPa)')
    ax.legend(title='Au-on-Elastomer')  # fontsize='x-small'
    ax.grid(alpha=0.25)
    # ax.set_title('4-wire Resistance vs. Thickness')
    plt.tight_layout()
    plt.savefig(join(PATH_SAVE, 'extend_Osmani_et_al_2016.png'), dpi=300, facecolor='w', bbox_inches='tight')
    plt.show()
