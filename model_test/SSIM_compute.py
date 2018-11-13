import math
import numpy as np
from scipy.signal import convolve2d

# you input image should be within [0, 255]

def round_sig(x, sig=2):
	return round(x, sig-int(math.floor(math.log10(abs(x))))-1)

# to imitate gaussian filter/ fspecial in Matlab
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
	m,n = [(ss-1.)/2. for ss in shape]
	y,x = np.ogrid[-m:m+1,-n:n+1]
	h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
	h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
	sumh = h.sum()
	if sumh != 0:
		h /= sumh
	window = h
	for i in range(window.shape[0]):
		for j in range(window.shape[1]):
			window[i,j] = round_sig(window[i,j], sig=4)
	return window

# image is target image, scaled_image is test image, others are deafult value here
def compute_ssim(image, scaled_image, K1= 0.01, K2= 0.03, L= 255.0):

	image_h = image*1.0		# this conversion is very important
	image_b = scaled_image*1.0

	window = matlab_style_gauss2D(shape=(11,11), sigma = 1.5)		# not exactly same here, should test your result on two softwares
	mu1 = convolve2d(image_h, np.rot90(window), mode='valid')		#print(np.max(mu1)), this is exactly what 'filter2' does in matlab 
	mu2 = convolve2d(image_b, np.rot90(window), mode='valid')		#print(np.max(mu2))

	mu1_sq = np.square(mu1)
	mu2_sq = np.square(mu2)
	mu1_mu2 = mu1*mu2
	num1 = convolve2d(image_h**2, np.rot90(window), mode='valid')
	sigma_mu1 =  num1 - mu1_sq		
	sigma_mu2 = convolve2d(np.square(image_b), np.rot90(window), mode='valid')-mu2_sq	#print(np.max(mu2))
	sigma12 = convolve2d(np.multiply(image_b, image_h), np.rot90(window), mode='valid')-mu1_mu2

	C1 = (K1*L)*(K1*L)
	C2 = (K2*L)*(K2*L)
	ssim_map = np.divide(np.multiply((2*mu1_mu2+C1),(2*sigma12+C2)) , np.multiply( (mu1_sq+mu2_sq+C1), (sigma_mu1+sigma_mu2+C2)  ) )
	return np.mean(ssim_map)
