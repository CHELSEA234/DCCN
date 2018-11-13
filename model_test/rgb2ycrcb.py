import numpy as np 

# this function can performance exactly as what Matlab does
def myRGB2YCRCB(image_array):
	image_array_flatten = np.reshape(image_array, (-1, 3))/255.0
	new_image_flatten = np.zeros((image_array_flatten.shape))
	new_image_flatten[:,0] = (image_array_flatten[:,0]*65.481+image_array_flatten[:,1]*128.553+image_array_flatten[:,2]*24.966)+16
	new_image_flatten[:,1] = (image_array_flatten[:,0]*(-37.797)+image_array_flatten[:,1]*(-74.203)+image_array_flatten[:,2]*112)+128
	new_image_flatten[:,2] = (image_array_flatten[:,0]*112+image_array_flatten[:,1]*(-93.786)+image_array_flatten[:,2]*(-18.214))+128
	new_image_flatten = np.array(new_image_flatten, dtype='uint8')
	new_image = np.reshape(new_image_flatten, image_array.shape)
	return new_image


def ycbcr2rgb(im):
	im = im.astype('float32')
	image_array_flatten = np.reshape(im, (-1, 3))
	new_image_flatten = np.zeros((image_array_flatten.shape))	
	new_image_flatten[:,0] = (image_array_flatten[:,0]-16)*1.16438355+(image_array_flatten[:,2]-128)*1.59602715 
	new_image_flatten[:,1] = (image_array_flatten[:,0]-16)*1.16438355-(image_array_flatten[:,1]-128)*0.3917616-(image_array_flatten[:,2]-128)*0.81296805
	new_image_flatten[:,2] = (image_array_flatten[:,0]-16)*1.16438355+(image_array_flatten[:,1]-128)*2.01723105		

	new_image_flatten[new_image_flatten<0] = 0
	new_image_flatten[new_image_flatten>255.] = 255.  

	new_image_flatten = np.array(new_image_flatten, dtype='uint8')
	new_image = np.reshape(new_image_flatten, im.shape)

	return new_image

