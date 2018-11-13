import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math
import scipy.io as sio
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="PyTorch LapSRN Test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model_epoch_50.pth", type=str, help="model path")
parser.add_argument("--image", default="baby_GT", type=str, help="image name")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model = torch.load(opt.model)["model"]

im_gt_y = sio.loadmat("Set5/" + opt.image + ".mat")['im_gt_y']
im_b_y = sio.loadmat("Set5/" + opt.image + ".mat")['im_b_y']
           
im_gt_y = im_gt_y.astype(float)
im_b_y = im_b_y.astype(float)      

psnr_bicubic = PSNR(im_gt_y, im_b_y, shave_border=opt.scale)

im_input = im_b_y/255.

im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

if cuda:
    model = model.module.cuda()
    im_input = im_input.cuda()
 
start_time = time.time()
out = model(im_input)
elapsed_time = time.time() - start_time

out = out.cpu()

im_h_y = out.data[0].numpy().astype(np.float32)

im_h_y = im_h_y*255.
im_h_y[im_h_y<0] = 0
im_h_y[im_h_y>255.] = 255.            

psnr_predicted = PSNR(im_gt_y, im_h_y[0,:,:],shave_border=opt.scale)


print("Scale=",opt.scale)
print("PSNR_predicted=", psnr_predicted)
print("PSNR_bicubic=", psnr_bicubic)
print("It takes {}s for processing".format(elapsed_time))
