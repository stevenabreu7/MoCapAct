

import h5py
import numpy as np

from IPython.display import HTML
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.iteritems():
        print("    %s: %s" % (key, val))


def visualize_images(images, show=True, to_file = None):
    fig, ax = plt.subplots()
    if len(images) > 0:
        x = ax.imshow(images[0])

        def update(frame):
            x.set_data(images[frame])
            return x

        ani = FuncAnimation(fig, update, frames=range(len(images)), interval=50)
        if to_file is not None:
            ani.save(to_file, writer='imagemagick', fps=30)
        if show:
            display(HTML(ani.to_html5_video()))
        plt.close()

file = h5py.File('./data/custom_rollouts_running/CMU_016_55.hdf5', 'r')

clip = 'CMU_016_55-0-300'
clip = 'CMU_016_55-0-149'
# print(file[f'{clip}/start_metrics'].keys())
# start_lens = np.array(file[f'{clip}/start_metrics/episode_lengths'][()])
# rsi_lens = np.array(file[f'{clip}/rsi_metrics/episode_lengths'][()])
# print(f"start {start_lens.mean():6.2f} +- {start_lens.std():6.2f}, range {start_lens.min():3d} - {start_lens.max():3d}")
# print(f"rsi   {rsi_lens.mean():6.2f} +- {rsi_lens.std():6.2f}, range {rsi_lens.min():3d} - {rsi_lens.max():3d}")
# file.visititems(print_attrs)
print(file)
ims = file[f'{clip}/0/images']

visualize_images(ims, to_file="./rollout.mp4")