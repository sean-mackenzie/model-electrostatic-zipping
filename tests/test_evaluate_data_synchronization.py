import os
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def visualize_staircase_levels(sampled_time, sampled_levels, leading_edge, slew_rate=0.001):
    # add time points to show V(t) in between current sampling times
    if leading_edge is True:
        t_steps = sampled_time[1:] - slew_rate
        l_steps = sampled_levels[:-1]
    else:
        t_steps = sampled_time[:-1] + slew_rate
        l_steps = sampled_levels[1:]
    # concat
    stair_time = np.concatenate((t_steps, sampled_time))
    stair_levels = np.concatenate((l_steps, sampled_levels))
    # sort by time
    stair_time, stair_levels = list(zip(*sorted(zip(stair_time, stair_levels))))

    stair_time = np.array(stair_time)
    stair_levels = np.array(stair_levels)

    return stair_time, stair_levels


if __name__ == "__main__":

    base_dir = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/zipper_paper/Testing/Zipper Actuation/03072025_W12-D1_C19-30pT_20+10nmAu'

    filepath_coords = join(base_dir, 'results/test-idpt_test-46/test_coords_t_c__cal_c_20+10nmAu_t_test-46.xlsx')
    fig_title = 'tid46_testVAR3_300V_500HzSQU_data'
    filepath_iv = join(base_dir, 'I-V/xlsx/{}.xlsx'.format(fig_title))
    save_id = fig_title

    df = pd.read_excel(filepath_coords)
    df['t'] = df['frame'] / 20  # frame rate = 20 Hz

    df_amplitude = pd.read_excel(filepath_iv, sheet_name='data_input')
    df_monitor = pd.read_excel(filepath_iv, sheet_name='data_output')

    # --- EVALUATE SYNCHRONIZATION (i.e., time delay)
    time_delay = -0.77
    df['t_sync'] = df['t'] + time_delay
    pid = 33
    df = df[df['id'] == pid]

    # visualize staircase voltage (only apply to AWG input voltage amplitude)
    pxi, pyi = 'dt', 'awg_volt'
    inp_t, inp_v = visualize_staircase_levels(
        sampled_time=df_amplitude[pxi], sampled_levels=df_amplitude[pyi],
        leading_edge=True, slew_rate=0.001,
    )

    monitor_delay = -0.3
    monitor_scale = 1.15
    df_monitor['TST_dt_sync'] = (df_monitor['TST'] + monitor_delay) * monitor_scale

    # --- plot
    px, py = 't_sync', 'z'
    jx, jy = 'dt', 'awg_volt'
    ix, iy = 'TST_dt_sync', 'MEAS_ZCOR'

    lw = 0.5
    ms = 3

    # --

    fig, (ax3, ax1, ax2) = plt.subplots(nrows=3, sharex=True, figsize=(10, 9))

    ax3.plot(df[px], df[py], 'r-', lw=lw, label='IDPT delay = {} s'.format(time_delay))
    ax3.set_ylabel('IDPT (shifted)', color='r')
    ax3.grid(alpha=0.25)
    ax3.legend(loc='upper right')

    ax3r = ax3.twinx()
    ax3r.plot(df_amplitude[jx], df_amplitude[jy], 'bo', ms=ms, label='Input: NO DELAY')
    ax3r.plot(inp_t, inp_v, 'b-', lw=lw, label='Input (continuous)')
    ax3r.set_ylabel('AWG Input Voltage (raw)', color='b')
    ax3r.legend(loc='lower right')

    # ---

    ax1.plot(df[px], df[py], 'r-', lw=lw * 2, label='IDPT delay = {} s'.format(time_delay))
    ax1.set_ylabel('IDPT (shifted)', color='r')
    ax1.grid(alpha=0.25)
    ax1.legend(loc='upper right')

    ax1r = ax1.twinx()
    ax1r.plot(df_monitor[ix], df_monitor[iy], 'k-', lw=lw / 1.5,
              label='Delay = {} s \n Scale = {}X'.format(monitor_delay, monitor_scale))
    ax1r.set_ylabel('Monitor (shifted + scaled)')
    ax1r.legend(loc='lower right', title='Monitor')

    # ---

    ax2.plot(df_amplitude[jx], df_amplitude[jy], 'bo', ms=ms, label='Input: NO DELAY')
    ax2.plot(inp_t, inp_v, 'b-', lw=lw, label='Input (continuous)')
    ax2.plot(inp_t, inp_v * -1, 'b-', lw=lw)
    ax2.set_ylabel('AWG Input Voltage (raw)', color='b')
    ax2.grid(alpha=0.25)
    ax2.legend(loc='upper right')

    ax2r = ax2.twinx()
    ax2r.plot(df_monitor[ix], df_monitor[iy], 'k-', lw=lw,
              label='Delay = {} s \n Scale = {}X'.format(monitor_delay, monitor_scale))
    ax2r.set_ylabel('Monitor (shifted + scaled)')
    ax2r.legend(loc='lower right', title='Monitor')

    ax2.set_xlabel('Mixed Time (s)')
    ax2.set_xlim([-0.75, 10])
    ax2.set_xticks(np.arange(-0.5, 10.1, 0.5))

    plt.suptitle(fig_title)
    plt.tight_layout()
    plt.savefig(join(base_dir, 'evaluate_{}_synchronization.png'.format(save_id)),
                dpi=300, facecolor='w', bbox_inches='tight')
    plt.show()