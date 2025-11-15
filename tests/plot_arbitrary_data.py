# plot_arbitrary_data.py

import os
from os.path import join
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from utils.energy import mechanical_energy_density_Gent, mechanical_energy_density_NeoHookean

if __name__ == "__main__":

    save_path = '/Users/mackenzie/Desktop'


    a = 16658
    b = 0.5
    x = np.linspace(10, 100, 20) * 1e-6
    y = a * x**b
    fig, ax = plt.subplots(figsize=(4.5, 3.75))
    ax.plot(x, y, '-', lw=1.25)
    ax.set_xlabel('Thickness (um)')
    ax.set_ylabel(r'$V_{PI} \: (V)$')
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


    plot_energy_density_vs_prestretch = False
    if plot_energy_density_vs_prestretch:
        E = 1.2e-6
        mu = E / 3  # shear modulus of Elastosil
        l = np.linspace(1.0001, 1.4, 50)
        Jms = [15, 54]

        fig, ax = plt.subplots(figsize=(5.5, 3.75))
        for J in Jms:
            U_Gent = mechanical_energy_density_Gent(mu, J, l)
            ax.plot(l, U_Gent, '-', lw=1, label=f'Gent: {J}')

        U_NH = mechanical_energy_density_NeoHookean(mu, l)
        ax.plot(l, U_NH, '--', lw=1, label=f'Neo-Hookean')

        ax.set_xlabel('Pre-stretch')
        ax.set_ylabel('Energy Density (J/m$^3$)')
        ax.grid(alpha=0.25)
        ax.legend(title='Jm')
        plt.tight_layout()
        plt.savefig(join(save_path, 'energy_density_comparison.png'),
                    dpi=300, facecolor='w', bbox_inches='tight')
        plt.show()

    # --


    plot_osmani = False
    if plot_osmani:
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
