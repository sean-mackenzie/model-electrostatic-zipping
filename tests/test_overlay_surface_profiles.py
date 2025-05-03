from os.path import join
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.settings import read_settings_to_dict_handler
from utils.empirical import read_surface_profile


if __name__ == "__main__":

    # THESE ARE THE ONLY SETTINGS YOU SHOULD CHANGE
    ROOT_DIR = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024'
    BASE_DIR = join(ROOT_DIR, 'zipper_paper/Testing/Zipper Actuation')

    MEMB_ID = 'C17-20pT'
    TEST_CONFIGS = [x for x in os.listdir(BASE_DIR) if x.endswith('_' + MEMB_ID)]
    WIDS = [x.split('_W')[1].split('-')[0] for x in TEST_CONFIGS]

    # --- read all surface profiles for this membrane
    read_profiles = False  # True False
    if read_profiles:

        surf_profile_subset = 'full'
        include_hole = True

        df = []
        for TEST_CONFIG, WID in zip(TEST_CONFIGS, WIDS):
            # ---
            # directories
            SAVE_DIR = join(BASE_DIR, TEST_CONFIG, 'analyses')
            SAVE_SETTINGS = join(SAVE_DIR, 'settings')
            SAVE_COORDS = join(SAVE_DIR, 'coords')
            PATH_REPRESENTATIVE = join(SAVE_DIR, 'representative_test{}')

            # -
            # settings
            FP_SETTINGS = join(SAVE_SETTINGS, 'dict_settings.xlsx')
            dict_settings = read_settings_to_dict_handler(filepath=FP_SETTINGS, name='settings', update_dependent=False)

            if 'fid_process_profile' in dict_settings.keys():
                surf_fid_override = dict_settings['fid_process_profile']
            else:
                surf_fid_override = None

            df_surface = read_surface_profile(
                dict_settings,
                subset=surf_profile_subset,
                hole=include_hole,
                fid_override=surf_fid_override,
            )

            df_surface.insert(0, 'fid_process_profile', dict_settings['fid_process_profile'])
            df_surface.insert(0, 'fid', dict_settings['fid'])
            df_surface.insert(0, 'wid', WID)

            df.append(df_surface)

        df = pd.concat(df)
        df.to_excel(join(BASE_DIR, MEMB_ID + '_surface_profiles.xlsx'), index=False)
    else:
        df = pd.read_excel(join(BASE_DIR, MEMB_ID + '_surface_profiles.xlsx'))

    # -

    # --- plot surface profiles overlay

    overlay_all = False
    if overlay_all:
        wids = df['wid'].unique()
        wids.sort()

        fig, ax = plt.subplots(figsize=(4, 2.75))
        for wid in wids:
            df_wid = df[df['wid'] == wid]
            ax.plot(df_wid['r'], df_wid['z'], '.', ms=1, label=wid)
        ax.set_xlabel(r'$r \: (\mu m)$')
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.grid(alpha=0.125)
        ax.legend(fontsize='x-small', loc='lower left',
                  title=r'$W_{ID}$', title_fontsize='small',
                  markerscale=3, handletextpad=0.6, labelspacing=0.25)
        plt.tight_layout()
        plt.savefig(join(BASE_DIR, MEMB_ID + '_surface_profiles.png'),
                    dpi=300, facecolor='w', bbox_inches='tight')
        plt.show()
        plt.close()

    overlay_by_aspect_ratio = False
    if overlay_by_aspect_ratio:
        wid_groups = [[11, 12], [13]]
        clrs = ['k', 'r', 'b']

        fig, axes = plt.subplots(ncols=len(wid_groups), sharey=True, figsize=(3.75 * len(wid_groups), 2.75))
        for ax, wids in zip(axes, wid_groups):
            for wid, clr in zip(wids, clrs):
                df_wid = df[df['wid'] == wid]
                ax.plot(df_wid['r'], df_wid['z'], '.', color=clr, ms=1, label=wid)
            ax.set_xlabel(r'$r \: (\mu m)$')
            ax.grid(alpha=0.125)
            ax.legend(fontsize='x-small', loc='lower left',
                      title=r'$W_{ID}$', title_fontsize='small',
                      markerscale=3, handletextpad=0.6, labelspacing=0.25)
        axes[0].set_ylabel(r'$z \: (\mu m)$')
        plt.tight_layout()
        plt.savefig(join(BASE_DIR, MEMB_ID + '_surface_profiles_by_AR.png'),
                    dpi=300, facecolor='w', bbox_inches='tight')
        plt.show()
        plt.close()


    # --

    a = 1

    print("Done.")