# -*- coding: utf-8 -*-
"""
Created on Tue May 03 16:32:53 2016

@author: Peter
"""

import numpy as np
import openslide

def open_image(slide):
    """
    Open openslide image
    """
    if isinstance(slide, str):
        slide = openslide.open_slide(slide)
    return slide

def get_image(slide, para, numpy=True):
    """
    Returns cropped image given a set of parameters.
    You can feed a string or an openslide image.
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
    
def get_x_y(slide, point_l, level):
    """
    Given a point (x_l, y_l) at a certain level. This function
    will return the coordinates associated to level 0 of this point (x_0, y_0).
    """
    x_l, y_l = point_l
    size_x_l = slide.level_dimensions[level][0]
    size_y_l = slide.level_dimensions[level][1]
    size_x_0 = float(slide.level_dimensions[0][0])
    size_y_0 = float(slide.level_dimensions[0][1])
  
    x_0 = x_l * size_x_0 / size_x_l
    y_0 = y_l * size_y_0 / size_y_l
    point_0 = (int(x_0), int(y_0))
    return point_0

def get_x_y_from_0(slide, point_0, level):
    """
    Given a point (x0, y0) at level 0, this function will return 
    the coordinates associated to the level 'level' of this point (x_l, y_l).
    """
    x_0, y_0 = point_0
    size_x_l = slide.level_dimensions[level][0]
    size_y_l = slide.level_dimensions[level][1]
    size_x_0 = float(slide.level_dimensions[0][0])
    size_y_0 = float(slide.level_dimensions[0][1])
  
    x_l = x_0 * size_x_l / size_x_0
    y_l = y_0 * size_y_l / size_y_0
  
    point_l = (int(x_l), int(y_l))
    return point_l
                      
def get_size(slide, size_from, level_from, level_to, round_scale=True):
    """
    Given a size at a certain level, this function will return
    this same size but at a different level.
    """
    size_x, size_y = size_from
    downsamples = slide.level_downsamples
    scal = float(downsamples[level_from]) / downsamples[level_to]
    if round_scale:
        func_round = round
    else:
        func_round = lambda x: x
    size_x_new = func_round(float(size_x) * scal)
    size_y_new = func_round(float(size_y) * scal)
    size_to = size_x_new, size_y_new
    return size_to

def find_square(slide, point, current_level, final_level, nber_pixels):
    """
    For a given pixel, returns a square centered on this pixel of a certain h and w
    """
    x_0, y_0 = get_x_y(slide, point, current_level)
    if isinstance(nber_pixels, int):
        height = int(np.ceil(np.sqrt(nber_pixels)))
        size_from = (height, height)
    elif isinstance(nber_pixels, tuple):
        size_from = nber_pixels
    else:
        raise NameError("Issue number 0002, nber_pixels should be of type int or tuple")
    if final_level == 0:
        size_final_level = size_from
    else:
        size_final_level = get_size(slide, size_from, final_level, 0)

    new_x = max(x_0 - size_final_level[0] / 2, 0)
    new_y = max(y_0 - size_final_level[1] / 2, 0)
    return_list = [new_y, new_x, size_final_level[0], size_final_level[1], final_level]
    return_list = [int(el) for el in return_list]
    return return_list


def white_percentage(rgb, white_thresh=220, tolerance=0.8):
    """
    Given an rgb image, computes a white score and return 
    true or false depending if this satisfies the condition.
    """
    score = (rgb.mean(axis=2) > white_thresh).sum()
    score = score / (rgb.shape[0] * rgb.shape[1])
    good_to_go = score < tolerance
    return good_to_go
