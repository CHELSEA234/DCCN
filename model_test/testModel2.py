import sys
sys.path.append(r'/auto/rcf-proj2/jc/xiaoguo/SISR/Helper')
import argparse
import torch
from torch.autograd import Variable
from rgb2ycrcb import myRGB2YCRCB
from PIL import Image
from scipy import misc
from PSNR_compute import PSNRLoss as PSNR
from SSIM_compute import compute_ssim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def test_phase(model, scale, dataset='Set5'):

	if dataset == 'Set5':
		image_name =['butterfly', 'head', 'woman', 'baby', 'bird']
		# image_name =['butterfly']

	psnr_value = ssim_value = 0
	single_psnr_value = single_ssim_value = 0

	for i in range(len(image_name)):

		im_gt_y = sio.loadmat("Set5_x2/" + image_name[i] + "_x2.mat")['im_gt_y']
		im_b_y = sio.loadmat("Set5_x2/" + image_name[i] + "_x2.mat")['im_b_y']
		           
		im_gt_y = im_gt_y.astype(float)
		im_b_y = im_b_y.astype(float)

		width = im_gt_y.shape[0]-int(im_gt_y.shape[0]%scale)
		height = im_gt_y.shape[1]-int(im_gt_y.shape[1]%scale)
		im_gt_y = im_gt_y[:width, :height]
		im_b_y = im_b_y[:width, :height]

		im_input = im_b_y/255.

		im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

		im_input = im_input.cuda()
		out = model(im_input)
		out = out.cpu()
		im_h_y = out.data[0].numpy().astype('float32')
		im_h_y = im_h_y[0,:,:]	

		im_h_y = im_h_y*255.
		im_h_y[im_h_y<0] = 0
		im_h_y[im_h_y>255.] = 255.  

		im_gt_y = im_gt_y[int(scale): width-int(scale), int(scale): height-int(scale)]
		im_h_y = im_h_y[int(scale): width-int(scale), int(scale): height-int(scale)]

		single_psnr_value = PSNR(im_h_y, im_gt_y)
		single_ssim_value = compute_ssim(im_h_y, im_gt_y)
		psnr_value = psnr_value+ single_psnr_value
		ssim_value = ssim_value+ single_ssim_value
	return psnr_value/len(image_name), ssim_value/len(image_name)

model = torch.load('model_epoch_50.pth')["model"]
psnr_value, ssim_value = test_phase(model, scale=2)
print ('''''''''''''''''''''''')
print ('psnr and ssim values are', psnr_value, ssim_value)
print ('''''''''''''''''''''''')

# model = torch.load('model_epoch_6_iteration_9600.pth')["model"]
# psnr_value, ssim_value = test_phase(model, scale=2)
# print ('''''''''''''''''''''''')
# print ('psnr and ssim values are', psnr_value, ssim_value)
# print ('''''''''''''''''''''''')
# model = torch.load('model_epoch_6_iteration_17364.pth')["model"]
# psnr_value, ssim_value = test_phase(model, scale=2)
# print ('''''''''''''''''''''''')
# print ('psnr and ssim values are', psnr_value, ssim_value)
# print ('''''''''''''''''''''''')
# model = torch.load('model_epoch_7_iteration_9600.pth')["model"]
# psnr_value, ssim_value = test_phase(model, scale=2)
# print ('''''''''''''''''''''''')
# print ('psnr and ssim values are', psnr_value, ssim_value)
# print ('''''''''''''''''''''''')
# model = torch.load('model_epoch_7_iteration_17364.pth')["model"]
# psnr_value, ssim_value = test_phase(model, scale=2)
# print ('''''''''''''''''''''''')
# print ('psnr and ssim values are', psnr_value, ssim_value)
# print ('''''''''''''''''''''''')
# model = torch.load('model_epoch_8_iteration_9600.pth')["model"]
# psnr_value, ssim_value = test_phase(model, scale=2)
# print ('''''''''''''''''''''''')
# print ('psnr and ssim values are', psnr_value, ssim_value)
# print ('''''''''''''''''''''''')
