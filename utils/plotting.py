# tests/test_model_sweep.py

# imports
from os.path import join
import matplotlib.pyplot as plt

from utils.shapes import complete_erf_profile



def plot_sweep_z_by_v(df_roots, path_save, save_id):
    mrkrs = ['b-', 'r--', 'g-.', 'k-']

    fig, ax = plt.subplots(figsize=(4.5, 3))

    for j in range(len(df_roots)):
        ax.plot(df_roots[j].U, df_roots[j].z * 1e6, mrkrs[j], label=vs[j])

    ax.set_xlabel('V (V)', fontsize=14)
    ax.set_ylabel('z (um)', fontsize=14)
    ax.legend(title=k)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, save_id + '_sweep_z-by-v.png'), dpi=300, facecolor='w')
    else:
        plt.show()
    plt.close()



def plot_profile_fullwidth(full, use, shape_function, width, depth, num_segments, shape_x0, dia_flat,
                           ax=None, ylbl=True, ls=None):
    """ ax = plot_profile_fullwidth(full, use, shape_function, width, depth, num_segments, shape_x0, dia_flat, ax) """
    # px, py = get_erf_profile(X, Z, num_segments, shape_x0, dia_flat)
    px, py = shape_function(width, depth, num_segments, shape_x0, dia_flat)

    # plot

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    if full is True:
        if use == 'erf':
            px_revolved, py_revolved = complete_erf_profile(px, py, width, dia_flat)
        elif use == 'flat':
            raise ValueError("Not implemented yet.")
            px_revolved, py_revolved = complete_flat_profile(px, py, width, theta=shape_x0)
        else:
            raise ValueError('must be: [erf, flat]')

        if ls is not None:
            ax.plot(px * 1e3, py * 1e6, ls)
            ax.plot(px_revolved * 1e3, py_revolved * 1e6, ls)
        else:
            ax.plot(px * 1e3, py * 1e6, ls='-', color='tab:blue')
            ax.plot(px_revolved * 1e3, py_revolved * 1e6, ls='-', color='tab:blue', label='Cross-section')
    else:
        ax.plot(px * 1e3, py * 1e6, ls='-', color='tab:blue', label='Zipping Path')
        ax.set_title('Dia.={}, Dia. Flat={}'.format(X, dia_flat))

    if ylbl:
        ax.set_ylabel('z (um)')

    ax.grid(alpha=0.25)
    ax.set_xlabel('r (mm)')

    if ls is None:
        ax.legend()

    if ax is None:
        plt.tight_layout()
        plt.show()
        plt.close()
    else:
        return ax


def plot_sweep_z_by_v_and_rz_profile():
    mrkrs = ['b-', 'r--', 'g-.', 'k-']

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(8.5, 3))

    for j in range(len(df_roots)):
        vval = vs[j]
        ls = mrkrs[j]

        ax2 = plot_profile_fullwidth(full=True,
                                     use='erf',
                                     shape_function=shape_function,
                                     width=vval,
                                     depth=Z,
                                     num_segments=num_segments,
                                     shape_x0=shape_x0,
                                     dia_flat=dia_flat,
                                     ax=ax2,
                                     ylbl=False,
                                     ls=ls,
                                     )

        ax1.plot(df_roots[j].U, df_roots[j].z * -1e6, ls, label=vs[j])

    ax2.set_xlabel('r (mm)', fontsize=14)
    ax2.set_title("x0={}, flat={}".format(shape_x0, dia_flat * 1e3))

    ax1.set_xlabel('V (V)', fontsize=14)
    ax1.set_ylabel('z (um)', fontsize=14)
    ax1.legend(title=k)
    ax1.grid(alpha=0.25)
    ax1.set_title('Pre-stretch={}'.format(pre_stretch))
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, save_id + '_sweep_z-by-v_and_rz-profile.png'), dpi=300, facecolor='w')
    else:
        plt.show()
    plt.close()


def plot_sweep_strain_by_z():

    # the below plotting method should be packaged into a function
    mrkrs = ['b-', 'r--', 'g-.', 'k-']

    # ---

    # plot

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(6, 5))

    for j in range(len(dfs)):

        df = dfs[j][0]
        vval = vs[j]
        ls = mrkrs[j]

        df['strain_xy'] = df['stretch_i'] / df['stretch_i'].iloc[0]
        df['strain_z'] = df['t_i'] / df['t_i'].iloc[0]
        df['strain_z_inv'] = 1 / df['strain_z']

        if export_excel_strain:
            df.to_excel(join(path_save, save_id + '_strain-by-z_{}_{}.xlsx'.format(k, vval)))

        # plot

        ax1.plot(df['dZ'] * 1e6, df['t_i'] * 1e6, ls, label=vval)

        if ls == 'b-':
            ax2.plot(df['dZ'] * 1e6, df['strain_xy'], color=ls[0], ls='-', label='in-plane')
            ax2.plot(df['dZ'] * 1e6, df['strain_z_inv'], color=ls[0], ls='--', label='out-of-plane')
        else:
            ax2.plot(df['dZ'] * 1e6, df['strain_xy'], color=ls[0], ls='-')
            ax2.plot(df['dZ'] * 1e6, df['strain_z_inv'], color=ls[0], ls='--')

        # -

    ax1.set_ylabel('Memb. Thickness', fontsize=14)
    ax1.grid(alpha=0.25)
    ax1.legend(title=k)

    ax2.set_ylabel('Strain', fontsize=14)
    ax2.set_xlabel('z (um)', fontsize=14)
    ax2.grid(alpha=0.25)
    ax2.legend()

    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, save_id + '_strain-by-z.png'), dpi=300, facecolor='w')
    else:
        plt.show()
    plt.close()