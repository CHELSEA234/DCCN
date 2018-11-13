import h5py
import numpy as np
import data_augmentation as DA
from PIL import Image
from scipy import misc

def pair_data_loading(dataset, c_dim, scale, method='bicubic', repeat=True, repeat_num=5):
	np.random.seed(0)
	order_label_dataset = dataset
	if repeat == True:
		order_label_dataset = np.repeat(order_label_dataset, repeat_num, axis= 0)
		print ('the sample has been repeated', repeat_num,'times')

	sample_num = len(order_label_dataset)

	height = order_label_dataset.shape[1]
	width = order_label_dataset.shape[2]
	height = height - (height%scale)
	width = width - (width%scale)
	label_size = image_size = width

	label_dataset = np.zeros(shape=(sample_num, label_size, label_size), dtype=np.uint8)
	training_dataset = np.zeros(shape=(sample_num, image_size, image_size, c_dim), dtype=np.uint8)

	arr = np.random.permutation(sample_num)     # randomly shuffling the data
	for index in range(arr.shape[0]):
		image_array = order_label_dataset[arr[index]]
		label_dataset[index] = image_array[:image_size, :image_size]

	label_dataset = DA.trainSet_change(label_dataset, label_size, c_dim)
	for i in range(sample_num):
		low_res_array = misc.imresize(misc.imresize(label_dataset[i], 1./scale, 'bicubic'), scale*1., method)
		training_dataset[i] = np.reshape(low_res_array, (image_size, image_size, c_dim))
		if (i+1) % 2000 == 0:
			print ('data processing is', i)
	label_dataset = np.reshape(label_dataset, (-1, label_size, label_size, 1))
	training_dataset = training_dataset

	return training_dataset, label_dataset

def data_loading(dataset, c_dim, repeat=True, repeat_num=5):
	np.random.seed(0)
	order_label_dataset = dataset
	if repeat == True:
		order_label_dataset = np.repeat(order_label_dataset, repeat_num, axis= 0)
		print ('the sample has been repeated', repeat_num,'times')

	sample_num = len(order_label_dataset)
	image_size = order_label_dataset.shape[1]
	training_dataset = np.zeros(shape=(sample_num, image_size, image_size), dtype=np.uint8)
	arr = np.random.permutation(sample_num)     # randomly shuffling the data
	for index in range(arr.shape[0]):
		image_array = order_label_dataset[arr[index]]
		training_dataset[index] = image_array[:image_size, :image_size]
	training_dataset = DA.trainSet_change(training_dataset, image_size, c_dim)
	training_dataset = np.reshape(training_dataset, (-1, image_size, image_size, 1))
	return training_dataset

def pair_data_making(dataset, c_dim, scale, method='bicubic', repeat=True, repeat_num=5):
	np.random.seed(0)
	order_label_dataset = dataset
	if repeat == True:
		order_label_dataset = np.repeat(order_label_dataset, repeat_num, axis= 0)
		print ('the sample has been repeated', repeat_num,'times')

	sample_num = len(order_label_dataset)

	height = order_label_dataset.shape[1]
	width = order_label_dataset.shape[2]
	height = height - (height%scale)
	width = width - (width%scale)
	label_size = width
	image_size = int(width/scale)

	print ('After processing, image_size and label_size are', image_size, label_size)

	label_dataset = np.zeros(shape=(sample_num, label_size, label_size), dtype=np.uint8)
	low_output_dataset = np.zeros(shape=(sample_num, label_size, label_size, c_dim), dtype=np.uint8)
	training_dataset = np.zeros(shape=(sample_num, image_size, image_size, c_dim), dtype=np.uint8)

	arr = np.random.permutation(sample_num)     # randomly shuffling the data
	for index in range(arr.shape[0]):
		image_array = order_label_dataset[arr[index]]
		label_dataset[index] = image_array[:label_size, :label_size]

	label_dataset = DA.trainSet_change(label_dataset, label_size, c_dim)
	for i in range(sample_num):
		low_res_array = misc.imresize(label_dataset[i], 1./scale, 'bicubic')
		low_output_array = misc.imresize(low_res_array, scale*1., 'bicubic')
		training_dataset[i] = np.reshape(low_res_array, (image_size, image_size, c_dim))
		low_output_dataset[i] = np.reshape(low_output_array, (label_size, label_size, c_dim))

		if (i+1) % 2000 == 0:
			print ('data processing is', i)
	print ('out of loop')
	label_dataset = np.reshape(label_dataset, (-1, label_size, label_size, 1))
	training_dataset = training_dataset
	low_output_dataset = low_output_dataset

	return training_dataset, label_dataset, low_output_dataset

# data = h5py.File('128_128_ycrcb.h5').get('training_dataset')
# training_dataset, label_dataset = pair_data_making(data, c_dim=1, scale=3, repeat_num=1)
# print (data.shape)
# print (label_dataset.shape)
# num = 151
# input_array = np.reshape(training_dataset[num], (126,126))
# x2_label_array = np.reshape(label_dataset[num], (126,126))

# import matplotlib.pyplot as plt
# plt.subplot(1,2,1)
# plt.imshow(input_array)
# plt.subplot(1,2,2)
# plt.imshow(x2_label_array)
# plt.show()
