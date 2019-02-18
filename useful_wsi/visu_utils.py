# -*- coding: utf-8 -*-
"""
Created on Tue May 03 16:32:53 2016

@author: Peter
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from .utils import open_image, get_whole_image, get_x_y_from_0, get_size

PLOT_ARGS = {'color': 'red', 'size': (12, 12), 'title': ""}

def visualise_cut(slide, list_pos, res_to_view=None, plot_args=PLOT_ARGS):
    """
    Plots the patches you are going to extract from the slide. So that they
    appear as red boxes on the lower resolution of the slide.
    Args:
        slide : str or openslide object.
        list_pos : list of parameters to extract tiles from slide.
        res_to_view : integer (default: None) resolution at which to
                      view the patch extraction.
        plot_args : dictionnary for any plotting argument.
    """
    slide = open_image(slide)
    if level is None:
        level = slide.level_count - 1
    elif level > slide.level_count - 1:
        print(" level ask is too low... It was setted accordingly")
        level = slide.level_count - 1
    whole_slide = get_whole_image(slide, res_to_view, numpy=True)
    fig = plt.figure(figsize=plot_args['size'])
    axes = fig.add_subplot(111, aspect='equal')
    axes.imshow(whole_slide)
    for para in list_pos:
        top_left_x, top_left_y = get_x_y_from_0(slide, (para[0], para[1]), res_to_view)
        width, height = get_size(slide, (para[2], para[3]), para[4], res_to_view)
        plot_seed = (top_left_x, top_left_y)
        patch = patches.Rectangle(plot_seed, width, height,
                                  fill=False, edgecolor=plot_args['color'])
        axes.add_patch(patch)
    axes.set_title(plot_args['title'], size=20)
    axes.axis('off')
    plt.show()
