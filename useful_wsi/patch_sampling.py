# -*- coding: utf-8 -*-
"""
Code for sampling from WSI

"""

import itertools
import numpy as np

from .tissue_segmentation import roi_binary_mask
from .utils import (find_square, get_size, get_whole_image,
                    get_x_y, get_x_y_from_0, mask_percentage, open_image)


def pj_slice(array_np, point_0, point_1=None):
    """
    Allows to slice numpy array's given one point or 
    two points.
    Args:
        array_np : Numpy array to slice
        point_0 : A tuple, or tuple like object of size 2 
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


def sample_patch_from_wsi(slide, mask=None, mask_level=None, 
                          patch_size=(512, 512), analyse_level=0,
                          list_func=[]):
    """
    Samples one tile from a slide where mask is 1

    Args:
        slide : String or open_slide object. The slide from which you wish to sample.
        mask : None by default or binary numpy array, where positive pixels correspond to tissue area and
               negative pixels to background areas in the slide. 
        mask_level : Integer or None. Level to which apply mask_function to the rgb 
                     image of the slide at that resolution. mask_function(slide[mask_level])
                     will return the binary image corresponding to the tissue.
        patch_size : Tuple of integers or None. If none the default tile size will (512, 512).
        analyse_level : Integer. Level resolution to use for extracting the tiles.
        list_func : None or list of functions to apply to the tiles. Useful to filter the tiles
                    that are part of the tissue mask. Very useful if the tissue mask is bad and 
                    samples many white background tiles, in this case it is interesting to add 
                    a function to eliminate tiles that are too white, like the function white_percentage.
    Returns:
        A list of 5 parameters corresponding to: [x, y, size_x_level, size_y_level, level]
    """       
    slide = open_image(slide)
    if mask_level is None:
        mask_level = slide.level_count - 1
    slide_png = get_whole_image(slide, mask_level, numpy=True)
    size_l = get_size(slide, patch_size, analyse_level, mask_level)
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
        para = find_square(slide, point, mask_level, analyse_level, patch_size)
    else:
        para = None
    return para


def remove_sample_from_mask(slide, para, mask, mask_level):
    """
    Given a square patch and a mask, removes the patch from mask.
    So that it can't be choosen again for instance...
    Args:
        slide : String or open_slide object. The slide from which you wish to sample.
        para : A list of 5 integer parameters corresponding to: [x, y, size_x_level, size_y_level, level]
        mask : None by default or binary numpy array, where positive pixels correspond to tissue area and
               negative pixels to background areas in the slide. 
        mask_level : Integer or None. Level to which apply mask_function to the rgb 
                     image of the slide at that resolution. mask_function(slide[mask_level])
                     will return the binary image corresponding to the tissue.    Returns: 
        An updated version of mask where the tile given by para and mask_level
        is removed.
    """
    if para is not None:
        point_0 = (para[1], para[0])
        size_l = (para[2], para[3])
        analyse_level = para[4]
        point_mask_res = get_x_y_from_0(slide, point_0, mask_level)
        point_mask_res = np.array(point_mask_res)
        size_mask_res = get_size(slide, size_l, analyse_level, mask_level)
        size_mask_res = np.array(size_mask_res)
        start_point = np.array([point_mask_res - size_mask_res, (0, 0)]).min(axis=0)
        end_point = start_point + 2*size_mask_res
        mask[start_point[0]:end_point[0], start_point[1]:end_point[1]] = 0

    return mask


def random_wsi_sampling(n_samples, slide, mask=None,
                        mask_level=None, patch_size=(512, 512),
                        analyse_level=0, with_replacement=False,
                        list_func=[]):
    """
    Randomly generate patches from slide.

    Args:
        n_samples : Integer, number of tiles to extract from the slide with the 
                    sampling method "random_sampling".
        slide : String or open_slide object. The slide from which you wish to sample.
        mask : None by default or binary numpy array, where positive pixels correspond to tissue area and
               negative pixels to background areas in the slide. 
        mask_level : Integer or None. Level to which apply mask_function to the rgb 
                     image of the slide at that resolution. mask_function(slide[mask_level])
                     will return the binary image corresponding to the tissue.
        patch_size : Tuple of integers or None. If none the default tile size will (512, 512).
        analyse_level : Integer. Level resolution to use for extracting the tiles.
        with_replacement : Bool, default to False. Wether or not you can sample with replacement. 
                           Here, if True, we would remove the previous patches from the original 
                           mask at each iteration.
        list_func : None or list of functions to apply to the tiles. Useful to filter the tiles
                    that are part of the tissue mask. Very useful if the tissue mask is bad and 
                    samples many white background tiles, in this case it is interesting to add 
                    a function to eliminate tiles that are too white, like the function white_percentage.

    Returns:
        A list of 5 parameters corresponding to: [x, y, size_x_level, size_y_level, level]
    """    
    list_para = []
    if mask_level is None:
        mask_level = slide.level_count - 1
    slide_png = get_whole_image(slide, mask_level, numpy=True)
    if mask is None:
        mask = np.ones_like(slide_png)[:, :, 0]
    mask = mask.astype('bool')

    for _ in range(n_samples):
        para = sample_patch_from_wsi(slide, mask, mask_level, patch_size,
                                     analyse_level, list_func)
        if para is not None:
            list_para.append(para)
        if not with_replacement:
            mask = remove_sample_from_mask(slide, para, mask, mask_level)
            if mask.sum() == 0:
                break
    return list_para


def grid_blob(slide, point_start, point_end, patch_size,
              analyse_level):
    """
    Forms a uniform grid starting from the top left point point_start
    and finishes at point point_end of size patch_size at level analyse_level
    for the given slide.
    Args:
        slide : String or open_slide object. 
        point_start : Tuple like object of integers of size 2.
        point_end : Tuple like object of integers of size 2.
        patch_size : Tuple like object of integers of size 2.
        analse_level : Integer. Level resolution to use for extracting the tiles.
    Returns:
        List of coordinates of grid.
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
    Correct patch by shifting so that the whole square patch can fit.
    Args:
        coord : A tuple like object of size 2 with integers.
        slide : String or open_slide object.
        analyse_level : Integer. Level resolution to use for extracting the tiles.
        patch_size : Tuple of integers or None.
    Returns:
        A numpy array of size 2 corresponding to the corrected or not
        orignal coord.
    """
    max_dimensions = np.array(slide.dimensions)[::-1]

    if analyse_level != 0:
        patch_size_0 = get_size(slide, patch_size, analyse_level, 0)
    else:
        patch_size_0 = patch_size
    coord = np.array([coord, max_dimensions - 1 - patch_size_0]).min(axis=0)
    return coord

def check_patch(slide, mask, coord_grid_0, mask_level, 
                patch_size, analyse_level,
                list_func, mask_tolerance=0.5,
                allow_overlapping=False,
                margin=0):
    """
    Filters a list of possible coordinates with a set of filtering parameters.

    Args:
        slide : String or open_slide object. The slide from which you wish to sample.
        mask : Binary numpy array, where positive pixels correspond to tissue area and
               negative pixels to background areas in the slide.
        coord_grid_0 : List of list of two elements where each (sub) list can be described as 
                       possible coordinates for a possible tile at analyse_level.
        mask_level : Integer or None. Level to which apply mask_function to the rgb 
                     image of the slide at that resolution. mask_function(slide[mask_level])
                     will return the binary image corresponding to the tissue.
        patch_size : Tuple of integers or None. If none the default tile size will (512 + margin, 512 + margin).
        analyse_level : Integer. Level resolution to use for extracting the tiles.
        list_func : None or list of functions to apply to the tiles. Useful to filter the tiles
                    that are part of the tissue mask. Very useful if the tissue mask is bad and 
                    samples many white background tiles, in this case it is interesting to add 
                    a function to eliminate tiles that are too white, like the function white_percentage.
        mask_tolerance : Float between 0 and 1. A tile will be accepted if pixel_in_mask / total_pixel > value.
                         So if mask_tolerance = 1, only tiles that are completly within the mask are accepted.
        allow_overlapping : Bool. False by default, this parameter does not influence the 'overlapping' parameter
                            above. This parameter only influences the tiles on that reach the border. In particular
                            if the tile extends out of the boundaries of the slide, in this case if allow_overlapping
                            is set to true, it will correct the extracted tile by allowing it to overlappe with it's
                            neighbouring tile.
        margin : Integer. By default set to 0, number of pixels at resolution 0 to add
                 to patch_size on each side. (different to overlapping as this is at resolution 0)
    Returns:
        List of parameters where each parameter is a list of 5 elements
        [x, y, size_x_level, size_y_level, level]
    """
    slide_png = get_whole_image(slide, level=mask_level, numpy=True)
    assert slide_png.shape[0:2] == mask.shape[0:2], "Raise value, mask not of the right shape {}".format(mask.shape[0:2])
    shape_mask = np.array(mask.shape[0:2])
    parameters = []
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
                still_add = True
                if ((coord_l + radius) != point_cent_l).any():
                    # If the patch is going off the border
                    still_add = False
                    if allow_overlapping:
                        coord_0 = correct_patch(coord_0, slide, analyse_level, patch_size)
                        still_add = True
                if still_add:
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
    Returns a list of tiles from slide given a mask generating method
    and a sampling method
    Args:
        slide : String or open_slide object. The slide from which you wish to sample.
        seed : Integer or None. Seed value to use for setting numpy randomness.
        mask_level : Integer or None. Level to which apply mask_function to the rgb 
                     image of the slide at that resolution. mask_function(slide[mask_level])
                     will return the binary image corresponding to the tissue.
        mask_function : Function that returns a binary image of same size as input. 
                        Mask_function is applied in order to determine the tissue areas on 
                        the slide.
        sampling_method : String. Possible values are 'grid', 'random_patches' for the 
                          patch sampling method.
                          If grid, we apply a grid on the tissue and extra all the tiles that
                          overlap on the tissue mask.
                          If random_patches, the tiles will be sampled at random from the tissue 
                          mask until no more available tissue or that n_samples has been reached.
        analyse_level : Integer. Level resolution to use for extracting the tiles.
        patch_size : Tuple of integers or None. If none the default tile size will (512, 512).
        overlapping : Integer. By default set to 0, number of pixels at analyse level to add
                      to patch_size on each side.
        list_func : None or list of functions to apply to the tiles. Useful to filter the tiles
                    that are part of the tissue mask. Very useful if the tissue mask is bad and 
                    samples many white background tiles, in this case it is interesting to add 
                    a function to eliminate tiles that are too white, like the function white_percentage.
        mask_tolerance : Float between 0 and 1. A tile will be accepted if pixel_in_mask / total_pixel > value.
                         So if mask_tolerance = 1, only tiles that are completly within the mask are accepted.
        allow_overlapping : Bool. False by default, this parameter does not influence the 'overlapping' parameter
                            above. This parameter only influences the tiles on that reach the border. In particular
                            if the tile extends out of the boundaries of the slide, in this case if allow_overlapping
                            is set to true, it will correct the extracted tile by allowing it to overlappe with it's
                            neighbouring tile. Only taken into account for the method "grid".
        n_samples : Integer, default to 10, number of tiles to extract from the slide with the 
                    sampling method "random_sampling".
        with_replacement : Bool, default to False. Wether or not you can sample with replacement in the case
                           of random sampling.

    Returns:
        List of parameters where each parameter is a list of 5 elements
        [x, y, size_x_level, size_y_level, level]
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

        margin_mask_level = get_size(slide, (overlapping, 0),
                                     0, analyse_level)[0]
        parameter = check_patch(slide, wsi_tissue, wsi_mask, grid_coord,
                                mask_level, patch_size, analyse_level,
                                list_func, mask_tolerance=mask_tolerance,
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
