# -*- coding: utf-8 -*-
"""
Code for sampling from WSI

"""

import itertools
import numpy as np

from .tissue_segmentation import roi_binary_mask
from .utils import find_square, get_size, get_whole_image
from .utils import get_x_y, get_x_y_from_0, open_image


def pj_slice(array_np, point_0, point_1=None):
    """
    Allows to slice numpy array's given one point or 
    two points.
    Args:
        array_np : numpy array to slice
        point_0 : a tuple, or tuple like object of size 2 
                  with integers.
        point_1 : None (default) or a tuple, or tuple like 
                  object of size 2 with integers.
    Returns:
        If point_1 is None, returns array_np evaluated in point_0,
        else returns a slice of array_np between point_0 and point_1.
    """
    x_0, y_0 = point_0
    if point_1 is None:
        result = array_np[x_0, y_0]
    else:
        x_1, y_1 = point_1
        result = array_np[x_0:x_1, y_0:y_1]
    return result


def sample_patch_from_wsi(slide, slide_png, mask=None, mask_resolution=None, 
                          size_square=(512, 512), analyse_level=0,
                          list_func=[]):
    """
    Sample one image from a slide where mask is 1
    """
    slide = open_image(slide)
    if mask_resolution is None:
        mask_resolution = slide.level_count - 1
    size_l = get_size(slide, size_square, analyse_level, mask_resolution)
    size_l = np.array(size_l)
    if mask is None:
        mask = np.ones_like(slide_png)[:, :, 0]
    x_mask, y_mask = np.where(mask)
    indice = np.random.randint(len(x_mask))
    point = np.array([x_mask[indice], y_mask[indice]])
    sub_img = pj_slice(slide_png, point - size_l // 2, point + size_l // 2)
    criterias = []
    for function in list_func:
        criterias.append(function(sub_img))
    if all(criterias):
        para = find_square(slide, point, mask_resolution, analyse_level, size_square)
    else:
        para = None
    return para


def remove_sample_from_mask(slide, para, mask, mask_resolution):
    """
    Given a square patch and a mask, removes the patch from mask.
    So that it can't be choosen again for instance...
    """
    if para is not None:
        point_0 = (para[1], para[0])
        size_l = (para[2], para[3])
        analyse_level = para[4]
        point_mask_res = get_x_y_from_0(slide, point_0, mask_resolution)
        point_mask_res = np.array(point_mask_res)
        size_mask_res = get_size(slide, size_l, analyse_level, mask_resolution)
        size_mask_res = np.array(size_mask_res)
        start_point = np.array([point_mask_res - size_mask_res, (0, 0)]).min(axis=0)
        end_point = start_point + 2*size_mask_res
        mask[start_point[0]:end_point[0], start_point[1]:end_point[1]] = 0

    return mask


def random_wsi_sampling(n_samples, slide, mask=None,
                        mask_resolution=None, size_square=(512, 512),
                        analyse_level=0, with_replacement=False,
                        list_func=[]):
    """
    Randomly generate patches from slide.
    """
    list_para = []
    if mask_resolution is None:
        mask_resolution = slide.level_count - 1
    slide_png = get_whole_image(slide, mask_resolution, numpy=True)
    if mask is None:
        mask = np.ones_like(slide_png)[:, :, 0]
    mask = mask.astype('bool')
    initial_mask = mask.copy()

    for _ in range(n_samples):
        para = sample_patch_from_wsi(slide, slide_png, mask, mask_resolution, size_square,
                                     analyse_level, list_func)
        mask = remove_sample_from_mask(slide, para, mask, mask_resolution)
        if with_replacement:
            mask = initial_mask
        if para is not None:
            list_para.append(para)
        if mask.sum() == 0:
            break
    return list_para


def grid_blob(slide, point_start, point_end, patch_size,
              analyse_level):
    """
    Returns grid for blob
    """
    if analyse_level == 0:
        patch_size_0 = patch_size
    else:
        patch_size_0 = get_size(slide, patch_size, analyse_level, 0)
    size_x, size_y = patch_size_0
    list_x = range(point_start[0], point_end[0], size_x)
    list_y = range(point_start[1], point_end[1], size_y)
    return list(itertools.product(list_x, list_y))


def correct_patch(coord, slide, analyse_level, patch_size):
    """
    Correct patch by shifting so that the whole square patch can fit!
    """
    max_dimensions = np.array(slide.dimensions)[::-1]

    if analyse_level != 0:
        patch_size_0 = get_size(slide, patch_size, analyse_level, 0)
    else:
        patch_size_0 = patch_size
    coord = np.array([coord, max_dimensions - 1 - patch_size_0]).min(axis=0)
    return coord


def mask_percentage(mask, point, radius, tolerance):
    """
    Given a binary image and a point and a radius -> sub_img
    Computes a score to know how much of the  sub_img is covered
    by tissue region. Given a tolerance threshold this will return
    a boolean.
    """
    sub_mask = pj_slice(mask, point - radius, point + radius)
    score = sub_mask.sum() / (sub_mask.shape[0] * sub_mask.shape[1])
    accepted = score > tolerance
    return accepted


def check_patch(slide, slide_png, mask, coord_grid_0,
                mask_level, patch_size, analyse_level,
                list_func, tolerance=0.5,
                allow_overlapping=False,
                margin=0):
    shape_mask = np.array(mask.shape[0:2])
    parameters = []
    margin_mask_level = get_size(slide, (overlapping, 0),
                                 0, analyse_level)[0]
    patch_size_l = get_size(slide, patch_size, analyse_level, mask_level)
    radius = np.array([max(el // 2, 1) for el in patch_size_l])
    for coord_0 in coord_grid_0:
        coord_l = get_x_y_from_0(slide, coord_0, mask_level)
        # coord_0 = np.array(coord_0)[::-1]
        # coord_l = np.array(coord_l)[::-1]
        point_cent_l = [coord_l + radius, shape_mask - 1 - radius]
        point_cent_l = np.array(point_cent_l).min(axis=0)
        if mask_percentage(mask, point_cent_l, radius, tolerance): ## only checking center point
            criterias = []
            sub_img = pj_slice(slide_png, point_cent_l - radius, point_cent_l + radius)
            for function in list_func:
                criterias.append(function(sub_img))
            if all(criterias):
                if ((coord_l + radius) != point_cent_l).any():
                    if allow_overlapping:
                        coord_0 = correct_patch(coord_0, slide, analyse_level, patch_size)
                    sub_param = [coord_0[1] - margin, coord_0[0] - margin, \
                                 patch_size[0] + 2 * margin, patch_size[1] + 2 * margin, \
                                 analyse_level]
                    parameters.append(sub_param)
    return parameters


def patch_sampling(slide, seed=None, mask_level=None,
                   mask_function=roi_binary_mask, sampling_method=None,
                   analyse_level=0, patch_size=None, overlapping=0,
                   list_func=None, mask_tolerance=0.5, allow_overlapping=False,
                   n_samples=10, with_replacement=False):
    """
    Returns a list of patches from slide given a mask generating method
    and a sampling method
    """
    np.random.seed(seed)
    slide = open_image(slide)

    if patch_size is None:
        patch_size = (512, 512)
    if list_func is None:
        list_func = list()
    if mask_level is None:
        mask_level = slide.level_count - 1

    wsi_tissue = get_whole_image(slide, level=mask_level, numpy=True)
    wsi_mask = mask_function(wsi_tissue)

    if sampling_method == 'grid':  # grid is just grid_etienne with marge = 0
        min_row, min_col, max_row, max_col = 0, 0, *wsi_mask.shape
        point_start_l = min_row, min_col
        point_end_l = max_row, max_col
        point_start_0 = get_x_y(slide, point_start_l, mask_level)
        point_end_0 = get_x_y(slide, point_end_l, mask_level)
        grid_coord = grid_blob(slide, point_start_0, point_end_0, patch_size,
                               analyse_level)
        parameter = check_patch(slide, wsi_tissue, wsi_mask, grid_coord,
                                mask_level, patch_size, analyse_level,
                                list_func, tolerance=mask_tolerance,
                                allow_overlapping=allow_overlapping,
                                margin=margin_mask_level)
        return_list = parameter
    elif sampling_method == "random_patches":
        return_list = random_wsi_sampling(n_samples, slide, wsi_mask,
                                          mask_level, patch_size, analyse_level,
                                          with_replacement=with_replacement,
                                          list_func=list_func)
    elif sampling_method == "random_patches_with_border":
        raise NameError('sampling method random_patches_with_border is not yet implemented...')
    else:
        raise NameError('sampling method is unknown...')
    return return_list
