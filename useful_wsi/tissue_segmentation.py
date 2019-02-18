# -*- coding: utf-8 -*-
"""
Created on Tue May 03 16:32:53 2016

@author: Peter
A collection of functions that can be used to generate a mask for a given tissue.
These functions were fine tuned on a personnal data. They can be used at your own risk.
"""
import numpy as np

from skimage.morphology import disk, opening, closing, dilation, remove_small_objects
from skimage.filters import threshold_otsu
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes



def opening_(image, disk_size=7):
    # Computes an opening of size disk.
    inter_image = image.copy()
    inter_image = opening(inter_image, disk(disk_size))
    return inter_image


def tissue_thresh(image, thresh=None):
    # tissue is in black... Tries to find the best background
    if thresh is None:
        thresh = threshold_otsu(image)
    binary = image > thresh
    if binary.dtype == 'bool':
        binary = binary + 0
    return binary


def fill_hole(bin_image, invert=False):
    # files holes
    values = np.unique(bin_image)
    if len(values) > 2:
        print("Not binary image")
        return []
    background = min(values)
    bin_image -= background
    bin_image[bin_image > 0] = 1
    if invert:
        bin_image -= 1
        bin_image[bin_image < 0] = 1
    result = np.copy(bin_image)
    binary_fill_holes(bin_image, output=result)
    return result


def remove_border(image_bin, border=15):
    # removes borders
    neg_border = border * -1
    result = np.copy(image_bin)
    result[:, :] = 0
    result[border:neg_border, border:neg_border] = image_bin[
        border:neg_border, border:neg_border]
    return result


def remove_isolated_points(binary_image, thresh=100):
    # removing tiny areas...
    # pdb.set_trace()
    lbl = label(binary_image)
    lbl = remove_small_objects(lbl, thresh)
    binary = (lbl > 0).astype(int)
    return binary


def find_ticket(rgb_image, _3tuple=(80, 80, 80)):
    # Find the "black ticket on the images"
    temp_image_3 = np.copy(rgb_image)
    temp_image_3[:, :, :] = 0
    for i in range(3):
        temp_image_1 = np.zeros(shape=rgb_image.shape[0:2])
        temp_image_1[np.where(rgb_image[:, :, i] < _3tuple[i])] = 1
        temp_image_3[:, :, i] = temp_image_1

    temp_resultat = temp_image_3.sum(axis=2)

    temp_resultat[temp_resultat > 2] = 3
    temp_resultat[temp_resultat < 3] = 0
    temp_resultat[temp_resultat == 3] = 1
    
    #temp_resultat = Filling_holes_2(temp_resultat)
    temp_resultat = closing(temp_resultat, disk(20))
    temp_resultat = opening_(temp_resultat, 20)
    temp_resultat = remove_border(temp_resultat)
    return temp_resultat


def preprocessing(image, thresh=200, invert=True):
    inter = opening_(image)
    inter = tissue_thresh(inter, thresh)
    inter = fill_hole(inter, invert=invert)
    inter = remove_border(inter)
    res = remove_isolated_points(inter)
    return res

def combining(numpy_array):
    res = np.sum(numpy_array, axis=2)
    res[res > 0] = 1
    return res

def roi_binary_mask2(sample, size=5, ticket=(80, 80, 80)):
    #  very slow function at resolution 4
    preproc_res = np.copy(sample)

    for i in range(3):  # RGB
        # this one is painfully slow..
        preproc_res[:, :, i] = preprocessing(sample[:, :, i])
    res = combining(preproc_res)
    ticket = find_ticket(sample, ticket)
    res = res - ticket
    res[res > 0] = 1
    res[res < 0] = 0
    res = opening_(res, size)
    return res

def roi_binary_mask(sample, size=5):
    val = threshold_otsu(sample[:, :, 0])
    mask = (sample[:, :, 0] < val).astype(int)
    mask = dilation(mask, disk(size))
    mask = fill_hole(mask)
    mask = remove_isolated_points(mask)
    return mask
