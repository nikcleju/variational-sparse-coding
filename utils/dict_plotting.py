"""
Plotting utility functions for learned sparse dictionary.

@Filename    dict_plotting
@Author      Kion 
@Created     5/29/20
"""
import matplotlib.animation as animation
import numpy as np
from matplotlib import pyplot as plt

import cv2
from sklearn.preprocessing import minmax_scale

def arrange_dict_similar(dict, reference, normalize=True, scale=True):
    """Arrange dictionary atoms in same order as a reference dictionary, by similarity
    """

    def find_most_similar(x, dict, exclude=[]):
        distances = np.zeros((dict.shape[1])) + np.inf
        for i in range(dict.shape[1]):
            if i not in exclude:
                distances[i] = np.linalg.norm(x - dict[:,i], 2)
        
        return np.argmin(distances)

    # Preprocess dictionaries
    if normalize:
        dict /= np.sqrt(np.sum(dict ** 2, axis=0))
        reference /= np.sqrt(np.sum(reference ** 2, axis=0))
    if scale:
        for i in range(dict.shape[1]):
            dict[:,i] = minmax_scale(dict[:,i], feature_range=(0,255))
        for i in range(reference.shape[1]):
            reference[:,i] = minmax_scale(reference[:,i], feature_range=(0,255))
    # Prepare new dict
    new_dict = np.zeros_like(dict)

    selected = []
    for i in range(dict.shape[1]):
        # Find atom in reference most similar to this
        ref_i = find_most_similar(dict[:,i], reference, exclude=selected)

        # Place on position of reference atom
        new_dict[:,ref_i] = dict[:,i]

        # Exclude those already matched
        selected.append(ref_i)

    return new_dict

# Nic
def save_dict_fast(phi, filename, sort_by_norm=False):
    """
    Fast save figure for dictionary, using OpenCV instead of matplotlib.

    :param phi: Dictionary. Dimensions expected as pixels x num atoms
    :param filename: File to save
    :param sort_by_norm: Sort atoms by descending norm or not
    """
    dict_mag = np.argsort(-1*np.linalg.norm(phi, axis=0)) if sort_by_norm else range(phi.shape[1])
    num_atoms = phi.shape[1]
    patch_size = int(np.sqrt(phi.shape[0]))
    
    grid_size = int(np.sqrt(num_atoms))
    border_size = 2
    border_color = 0

    total_size = grid_size*(patch_size + border_size) - border_size
    total_image = np.zeros((total_size, total_size)) + border_color

    for i in range(num_atoms):
        grid_row  = i // grid_size
        grid_col  = i % grid_size

        row_start = grid_row * (patch_size + border_size)
        col_start = grid_col * (patch_size + border_size)        
        row_stop  = row_start + patch_size
        col_stop  = col_start + patch_size

        total_image[row_start:row_stop, col_start:col_stop] = \
            minmax_scale(phi[:, dict_mag[i]], feature_range=(0,255)).reshape(patch_size, patch_size)

    cv2.imwrite(filename, total_image)

def show_dict(phi, save_dir):
    """
    Create a figure for dictionary
    :param phi: Dictionary. Dimensions expected as pixels x num atoms
    """
    dict_mag = np.argsort(-1*np.linalg.norm(phi, axis=0))
    num_atoms = phi.shape[1]
    patch_size = int(np.sqrt(phi.shape[0]))
    fig = plt.figure(figsize=(12, 12))
    for i in range(num_atoms):
        plt.subplot(int(np.sqrt(num_atoms)), int(np.sqrt(num_atoms)), i + 1)
        dict_element = phi[:, dict_mag[i]].reshape(patch_size, patch_size)
        plt.imshow(dict_element, cmap='gray')
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.savefig(save_dir, bbox_inches='tight')
    plt.close()

def show_phi_vid(phi_list):
    """
    Creates an HTML5 video for a list of dictionaries
    :param phi_list: List of dictionaries. Dimensions expected as time x pixels x num dictionaries
    :return: Matplotlib animation object containing HTML5 video for dictionaries over time
    """
    fig = plt.figure(figsize=(12, 12))
    num_dictionaries = phi_list.shape[2]
    patch_size = int(np.sqrt(phi_list.shape[1]))

    ax_list = []
    for p in range(num_dictionaries):
        ax_list.append(fig.add_subplot(int(np.sqrt(num_dictionaries)), int(np.sqrt(num_dictionaries)), p + 1))

    ims = []
    for i in range(phi_list.shape[0]):
        phi_im = []
        title = plt.text(0.5, .90, "Epoch Number {}".format(i),
                         size=plt.rcParams["axes.titlesize"],
                         ha="center", transform=fig.transFigure, fontsize=20)
        phi_im.append(title)
        for p in range(num_dictionaries):
            dict_element = phi_list[i, :, p].reshape(patch_size, patch_size)
            im = ax_list[p].imshow(dict_element, cmap='gray', animated=True)
            phi_im.append(im)
        ims.append(phi_im)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=2000)
    plt.close()
    return ani
