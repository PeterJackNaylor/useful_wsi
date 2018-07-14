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
    Visulazise the patches you are going to extract from the image.
    """
    slide = open_image(slide)
    if res_to_view is None:
        res_to_view = slide.level_count - 1
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
