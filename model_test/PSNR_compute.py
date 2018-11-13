from PIL import Image
import math
import numpy as np
from rgb2ycrcb import myRGB2YCRCB

# your input image should be within [0, 255]
def psnr(true_image, pre_image):
    mse = np.mean( (pre_image*1.0 - true_image*1.0) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = np.max(pre_image)
    return 20 * math.log10(255 / math.sqrt(mse))

def PSNRLoss(y_true, y_pred):
    mse = np.mean(np.square(y_pred*1.0 - y_true*1.0))
    PSNR = 20* math.log10(255) - 10*math.log10(mse)
    return PSNR
    