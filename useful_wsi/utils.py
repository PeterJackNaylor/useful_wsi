# -*- coding: utf-8 -*-
"""
Created on Tue May 03 16:32:53 2016

@author: Peter
"""

import numpy as np
import openslide

def load_pred_cnn():
    import sys
    import useful_wsi as module
    from os.path import join
    path = module.__path__[0]
    path = join(path, "..", "example", "best_wsi_tissue_segmentation")
    sys.path.append(path)
    from tissue_prediction import pred_cnn
    return pred_cnn

def open_image(slide):
    """
    Open openslide image
    Args:
        slide : String or other object (hopefully openslide object).
    Returns: 
        If string, returns an openslide object else return the input.
    """
    if isinstance(slide, str):
        slide = openslide.open_slide(slide)
    return slide

def get_image(slide, para, numpy=True):
    """
    Returns cropped image given a set of parameters.
    You can feed a string or an openslide image.
    Args:
        slide : String or openslide object from which we extract.
        para : List of 5 integers corresponding to: [x, y, size_x_level, size_y_level, level]
        numpy : Boolean, by default True, wether or not to convert the output to numpy array instead
                of PIL image.
    Returns:
        A tile (or crop) from slide corresponding to para. It can be a numpy array
        or a PIL image.

    """
    if len(para) != 5:
        raise NameError("Not enough parameters...")
    slide = open_image(slide)
    slide = slide.read_region((para[0], para[1]), para[4], (para[2], para[3]))

    if numpy:
        slide = np.array(slide)[:, :, 0:3]
    return slide

def get_whole_image(slide, level=None, numpy=True):
    """
    Return whole image at a certain level.
    Args:
        slide : String or openslide object from which we extract.
        level : Integer, by default None. If None the value is set to
                the maximum level minus one of the slide. Level at which
                we extract.
        numpy : Boolean, by default True, wether or not to convert the output to numpy array instead
                of PIL image.
    Returns:
        A numpy array or PIL image corresponding the whole slide at a given
        level.
    """
    if isinstance(slide, str):
        slide = openslide.open_slide(slide)
    
    if level is None:
        level = slide.level_count - 1
    elif level > slide.level_count - 1:
        print(" level ask is too low... It was setted accordingly")
        level = slide.level_count - 1
    sample = slide.read_region((0, 0), level, slide.level_dimensions[level])
    if numpy:
        sample = np.array(sample)[:, :, 0:3]
    return sample
    
def get_x_y(slide, point_l, level, integer=True):
    """
    Given a point point_l = (x_l, y_l) at a certain level. This function
    will return the coordinates associated to level 0 of this point point_0 = (x_0, y_0).
    Args:
        slide : Openslide object from which we extract.
        point_l : A tuple, or tuple like object of size 2 with integers.
        level : Integer, level of the associated point.
        integer : Boolean, by default True. Wether or not to round
                  the output.
    Returns:
        A tuple corresponding to the converted coordinates, point_0.
    """
    x_l, y_l = point_l
    size_x_l = slide.level_dimensions[level][0]
    size_y_l = slide.level_dimensions[level][1]
    size_x_0 = float(slide.level_dimensions[0][0])
    size_y_0 = float(slide.level_dimensions[0][1])
  
    x_0 = x_l * size_x_0 / size_x_l
    y_0 = y_l * size_y_0 / size_y_l
    if integer:
        point_0 = (int(x_0), int(y_0))
    else:
        point_0 = (x_0, y_0)
    return point_0

def get_x_y_from_0(slide, point_0, level, integer=True):
    """
    Given a point point_0 = (x0, y0) at level 0, this function will return 
    the coordinates associated to the level 'level' of this point point_l = (x_l, y_l).
    Inverse function of get_x_y
    Args:
        slide : Openslide object from which we extract.
        point_0 : A tuple, or tuple like object of size 2 with integers.
        level : Integer, level to convert to.  
        integer : Boolean, by default True. Wether or not to round
                  the output.
    Returns:
        A tuple corresponding to the converted coordinates, point_l.
    """
    x_0, y_0 = point_0
    size_x_l = slide.level_dimensions[level][0]
    size_y_l = slide.level_dimensions[level][1]
    size_x_0 = float(slide.level_dimensions[0][0])
    size_y_0 = float(slide.level_dimensions[0][1])
  
    x_l = x_0 * size_x_l / size_x_0
    y_l = y_0 * size_y_l / size_y_0
    if integer:
        point_l = (round(x_l), round(y_l))
    else:
        point_l = (x_l, y_l)
    return point_l
                      
def get_size(slide, size_from, level_from, level_to, integer=True):
    """
    Given a size (size_from) at a certain level (level_from), this function will return
    a new size (size_to) but at a different level (level_to).
    Args:
        slide : Openslide object from which we extract.
        size_from : A tuple, or tuple like object of size 2 with integers.
        level_from : Integer, initial level.
        level_to : Integer, final level.
        integer : Boolean, by default True. Wether or not to round
                  the output.
        Returns:
            A tuple, or tuple like object of size 2 with integers corresponding 
            to the new size at level level_to. Or size_to.
    """
    size_x, size_y = size_from
    downsamples = slide.level_downsamples
    scal = float(downsamples[level_from]) / downsamples[level_to]
    if integer:
        func_round = round
    else:
        func_round = lambda x: x
    size_x_new = func_round(float(size_x) * scal)
    size_y_new = func_round(float(size_y) * scal)
    size_to = size_x_new, size_y_new
    return size_to

def find_square(slide, point, level_from, level_to, nber_pixels):
    """
    For a given pair of coordinates at level level_from, returns a 
    square centered on these coordinates at level level_to. 
    The square will be of height h and of width w where if nber_pixel is
    integer h = w = ceil(sqrt(nber_pixel)), if nber_pixel is tuple 
    h, w = nber_pixel.
    Args:
        slide : Openslide object from which we extract.
        point : A tuple, or tuple like object of size 2 with integers corresponding
                to the center of the patch. 
        level_from : Integer, initial level.
        level_to : Integer, final level.
        nber_pixels: If integer, h = w = ceil(sqrt(nber_pixel))
                     If tuple, h, w = nber_pixel

    Returns:
        List of 5 integers corresponding to: [x, y, size_x_level, size_y_level, level]
    """
    x_0, y_0 = get_x_y(slide, point, level_from)
    if isinstance(nber_pixels, int):
        height = int(np.ceil(np.sqrt(nber_pixels)))
        size_from = (height, height)
    elif isinstance(nber_pixels, tuple):
        size_from = nber_pixels
    else:
        raise NameError("Issue number 0002, nber_pixels should be of type int or tuple")
    if level_to == 0:
        size_level_to = size_from
    else:
        size_level_to = get_size(slide, size_from, level_to, 0)

    new_x = max(x_0 - size_level_to[0] / 2, 0)
    new_y = max(y_0 - size_level_to[1] / 2, 0)
    return_list = [new_y, new_x, size_level_to[0], size_level_to[1], level_to]
    return_list = [int(el) for el in return_list]
    return return_list


def white_percentage(rgb, white_thresh=220, tolerance=0.8):
    """
    Given an rgb image, computes a white score and return 
    true or false depending if this satisfies the condition.
    Args:
        rgb : 3D matrix where the last dimension corresponds to 
              the image channels.
        white_thresh : Integer, by default set to 220. This default value
                       is only valid if rgb is of type uint8.
        tolerance : A float between 0 and 1. By default 0.5.

    Returns:
        A boolean.
    """
    score = (rgb.mean(axis=-1) > white_thresh).sum()
    score = score / (rgb.shape[0] * rgb.shape[1])
    accepted = score < tolerance
    return accepted

def mask_percentage(mask, point, radius, mask_tolerance=0.5):
    """
    Given a binary image and a point and a radius -> sub_img
    Computes a score to know how much of the  sub_img is covered
    by tissue region. Given a tolerance threshold this will return
    a boolean.
    tolerance is mask_tolerance
    tolerance of 1 means that the entire image is in the mask area.
    tolerance of 0.1 means that the image has to overlap at least at 10%
              with the mask.
    Args:
        mask : Binary numpy array, where positive pixels correspond to tissue area and
               negative pixels to background areas in the slide.
        point : A tuple like object of size 2 
                with integers.
        radius : None (default) or a tuple, or tuple like 
                 object of size 2 with integers.
        tolerance : A float between 0 and 1. By default 0.5.
    Returns:
        A boolean.
    """
    sub_mask = pj_slice(mask, point - radius, point + radius)
    score = sub_mask.sum() / (sub_mask.shape[0] * sub_mask.shape[1])
    accepted = score > tolerance
    return accepted



