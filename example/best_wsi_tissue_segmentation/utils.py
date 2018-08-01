
import numpy as np
from skimage import img_as_bool
from skimage import measure as meas
from skimage.morphology import remove_small_holes, remove_small_objects, binary_dilation, disk
from skimage.transform import resize

from segmentation_net.utils import expend


def sliding_window(image, stepSize, windowSize):
    # slide a window across the imag
    for y in range(0, image.shape[0] - windowSize[0] + stepSize, stepSize):
        for x in range(0, image.shape[1] - windowSize[1] + stepSize, stepSize):
            # yield the current window
            res_img = image[y:y + windowSize[1], x:x + windowSize[0]]
            change = False
            if res_img.shape[0] != windowSize[1]:
                y = image.shape[0] - windowSize[1]
                change = True
            if res_img.shape[1] != windowSize[0]:
                x = image.shape[1] - windowSize[0]
                change = True
            if change:
                res_img = image[y:y + windowSize[1], x:x + windowSize[0]]
            yield (x, y, x + windowSize[0], y + windowSize[1], res_img)

def slidding_window_predict(img, mean, mod):
    if mean is not None:
        img = img - mean
    rgb_shape_x = img.shape[0] - 184
    rgb_shape_y = img.shape[1] - 184
    res = np.zeros(shape=(rgb_shape_x, rgb_shape_y), dtype='uint8')
    res_prob = np.zeros(shape=(rgb_shape_x, rgb_shape_y), dtype='uint8')
    for x, y, w_x, h_y, slide in sliding_window(img, 212, (396, 396)):
        tensor = np.expand_dims(slide, 0)
        feed_dict = {mod.input_node: tensor,
                     mod.is_training: False}
        tensors_names = ["predictions", "probability"]
        tensors_to_get = [mod.predictions, mod.probability]
        
        tensors_out = mod.sess.run(tensors_to_get,
                                   feed_dict=feed_dict)
        res[y:(y+212), x:(x+212)] = tensors_out[0][0]
        res_prob[y:(y+212), x:(x+212)] = tensors_out[1][0, :, :, 0]
    res[res > 0] = 255
    all_tensors = [res, res_prob]
    out_dic = {}
    for name, tens in zip(tensors_names, all_tensors):
        out_dic[name] = tens
    return out_dic
    
def unet_pred(img, mean, mod):
    prev_shape = img.shape[0:2]
    img = resize(img, (512, 512), order=0, 
                 preserve_range=True, mode='reflect', 
                 anti_aliasing=True).astype(img.dtype)
    unet_img = expend(img, 92, 92)
    #mask = model.predict(rgb, mean=mean_array)['predictions'].astype('uint8')
    dic_mask = slidding_window_predict(unet_img, mean, mod)
    dic_mask['predictions'][dic_mask['predictions'] > 0] = 255
    dic_mask['predictions'] = img_as_bool(resize(dic_mask['predictions'], prev_shape, order=0, 
                                          preserve_range=True, mode='reflect', 
                                          anti_aliasing=True))
    dic_mask['probability'] = resize(dic_mask['probability'], prev_shape, order=0, 
                                     preserve_range=True, mode='reflect', 
                                     anti_aliasing=True).astype(dic_mask['probability'].dtype)
    return dic_mask
      
def CleanPrediction(m, thresh=100, disk_size=3):
    img = meas.label(m.copy())
    img = remove_small_objects(img, min_size=thresh)
    img = remove_small_holes(img, thresh)
    ## just to take a bigger margin around the images
    img = binary_dilation(img, disk(disk_size))
    return img
