
import numpy as np
import os
from segmentation_net import UnetPadded
from utils import unet_pred, CleanPrediction

#LOG = os.path.abspath('./UnetPadded__0.001__4/')
LOG = os.path.join(os.path.dirname(__file__), 'UnetPadded__0.001__4/')
MODEL = UnetPadded(image_size=(212, 212), log=LOG, n_features=4)
MEAN = np.load(os.path.join(os.path.dirname(__file__), 'mean_file.npy'))


def pred_cnn(image, model=MODEL, mean=MEAN):
    unet_imag = unet_pred(image, mean, model)['predictions'].astype('uint8')
    result = CleanPrediction(unet_imag)
    return result


