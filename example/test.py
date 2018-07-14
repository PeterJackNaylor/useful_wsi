# -*- coding: utf-8 -*-
"""
Created on Tue May 03 16:32:53 2016

@author: Peter
"""

import openslide
from useful_wsi import patch_sampling, visualise_cut
from useful_wsi import roi_binary_mask, white_percentage

def check_for_white(img):
    """
    Checking for white
    220 is the threshold
    0.8 is the tolerance level
    """
    return white_percentage(img, 220, 0.8)

def main():
    OPTIONS_APPLYING_MASK = {'mask_level': 2, 'function': roi_binary_mask}
    OPTIONS_SAMPLING = {'method': "random_patches", 'analyse_level': 0, 'patch_size': (512, 512),
                        'overlapping': 0, 'list_func': [white_percentage], 'mask_tolerance': 0.3,
                        'allow_overlapping': False, 'n_samples': 100, 'with_replacement': False}


    file_name = "TCGA-CN-4739-01A-02-BS2.fc87f5db-d311-4734-a200-7c7d4885b274.svs"

    list_roi = patch_sampling(file_name, 
                              o_mask=OPTIONS_APPLYING_MASK, 
                              o_sampling=OPTIONS_SAMPLING)

    print('We have so many patches {}.'.format(len(list_roi)))

    visualise_cut(file_name, list_roi)

if __name__ == '__main__':
    main()
