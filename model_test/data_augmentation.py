import numpy as np 
from PIL import Image
from scipy import misc

def turn_vertically(image):
	width = image.shape[0]
	height = image.shape[1]
	new_image = np.zeros((width, height))

	for y in range(height):
		for x in range(width//2):
			left = image[x,y]
			right = image[width-x-1,y]
			temp = left
			new_image[x,y] = right
			new_image[width-x-1,y] = temp

	if (width//2)*2 == width-1:
		new_image[int(width//2), :] = image[int(width//2), :]
	return new_image

def turn_horizontally(image):
	width = image.shape[0]
	height = image.shape[1]
	new_image = np.zeros((width, height), dtype=np.uint8)

	for x in range(width):
		for y in range(height//2):
			up = image[x,y]
			down = image[x,height-y-1]
			temp = up
			new_image[x,y] = down
			new_image[x,height-y-1] = up

	if (height//2)*2 == height-1:
		new_image[:, int(height//2)] = image[:, int(height//2)]
	return new_image

def rotatedConversion(image, seed):
	np.random.seed(seed)
	angle = [0, 90, 180, 270]
	rand_angle = np.random.choice(angle)
	image = Image.fromarray(image)
	img = image.rotate(rand_angle)
	image = np.asarray(img)		## image to array
	return image

def data_augmentation(image, seed):
	np.random.seed(seed)
	flag = np.random.randint(4)
	image = resizing(image, seed)
	if flag == 0:
		image = turn_vertically(turn_horizontally(image))
	elif flag == 1:
		image = turn_horizontally(image)
	elif flag == 2:
		image = turn_vertically(image)
	else:
		image
	image = rotatedConversion(image, seed)

	return image

def trainSet_change(training_dataset, image_size, c_dim, seed=0):
	sample_num = len(training_dataset)
	for i in range(sample_num):
		training_dataset[i] = data_augmentation(np.reshape(training_dataset[i], (image_size, image_size)), seed=seed)
	return training_dataset

def resizing(image, seed):
	np.random.seed(seed)
	downsizes = [1,0.5]
	rand_downsize = np.random.choice(downsizes)
	if rand_downsize == 0.5:
		img_array = Image.fromarray(image)
		img = misc.imresize( misc.imresize(img_array, rand_downsize*1., 'bicubic'), 1./rand_downsize, 'bicubic')
		image = np.asarray(img)
	return image
