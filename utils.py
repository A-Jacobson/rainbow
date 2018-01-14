import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import display


def plot_state(state):
    """
    allows image plots to be updated with for loop in jupyter notebook
    """
    display.clear_output(wait=True)
    plt.imshow(state)
    display.display(plt.gcf(), transient=True)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.close()


def grid_from_state(state):
    """
    transforms stacked state into numpy array of four images
    """
    return np.hstack([np.pad(np.array(state)[:, :, i], 2, mode='constant') for i in range(4)])


def process_state(state):
    state = np.array(state, dtype='float32').transpose((2, 0, 1))  # change WHC to CWH for pytorch
    state /= 255.  # rescale 0-1
    state = torch.from_numpy(state).unsqueeze(0)  # add batch dim
    return state