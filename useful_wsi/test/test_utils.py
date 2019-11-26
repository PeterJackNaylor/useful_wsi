from __future__ import absolute_import

import pytest

import numpy as np
import skimage

from useful_wsi.utils import pj_slice, white_percentage, mask_percentage

def test_pj_slice():

    res = np.array([[7.,6.,5.],[8.,7.,6.]])

    point_1 = (3, 1)
    point_2 = (5, 4)

    test_array = np.zeros((5, 5))
    for i in range(5):
        test_array[i:] += 1
        test_array[:,:i] += 1

    mat = pj_slice(test_array, point_1, point_2)

    assert mat.tolist() == res.tolist(), "pj_slice not working"

def test_white_percentage():

    rgb = skimage.data.rocket()
    thresh = 0.0024114461358313815
    true = white_percentage(rgb, tolerance=thresh + 0.001)
    false = white_percentage(rgb, tolerance=thresh - 0.001)
    assert true == True, "white_percentage true assertation"
    assert false == False, "white_percentage false assertation"

def test_mask_percentage():
    # we are testing the anchor 3x3 window, where we have five zeros and four ones
    mask = np.array([[ 0,  0,  0,  0,  1],
                     [ 0,  0,  0,  1,  1],
                     [ 0,  0,  1,  1,  1],
                     [ 0,  0,  0,  1,  1],
                     [ 0,  0,  0,  0,  0]])
    point = np.array([2,2])
    radius = 1
    mask_tolerance = 5 / 8
    rejected = mask_percentage(mask, point, radius, mask_tolerance=mask_tolerance)
    assert rejected == False, "This tile should be rejected"

    point = np.array([2,3])
    accepted = mask_percentage(mask, point, radius, mask_tolerance=mask_tolerance)
    assert accepted == True, "This tile should be accepted"

