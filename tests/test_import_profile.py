from os.path import join
import matplotlib.pyplot as plt

from utils.empirical import dict_from_tck


if __name__ == '__main__':

    # general inputs
    base_dir = '/Users/mackenzie/Desktop/zipper_paper/Fabrication/grayscale'
    filepath_tck = 'w{}/results/profiles_tck/fid{}_tc_k=3.xlsx'

    # specific inputs
    wid = 14
    fid = 3
    depth = 195.1
    radius = 1771.6
    num_segments = 3500  # NOTE: this isn't necessarily the final number of solver segments
    fp_tck = join(base_dir, filepath_tck.format(wid, fid))

    dict_fid = dict_from_tck(wid, fid, depth, radius, num_segments, fp_tck=None)

    fig, ax = plt.subplots()
    ax.plot(dict_fid['r'], dict_fid['z'])
    plt.show()