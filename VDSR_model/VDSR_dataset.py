import sys
sys.path.append(r'/auto/rcf-proj2/jc/xiaoguo/SISR/Helper')
import torch.utils.data as data
import numpy as np
import torch
import h5py

class DatasetFromHdf5(data.Dataset):
    def __init__(self):
        super(DatasetFromHdf5, self).__init__()

        file_name_0 = r'/auto/rcf-proj2/jc/xiaoguo/SISR/train_dataset/VDSR/train_91.h5'
        file_name_1 = r'/auto/rcf-proj2/jc/xiaoguo/SISR/train_dataset/0_8/train_91.h5'

        h5f_handler_0 = h5py.File(file_name_0, 'r')
        h5f_handler_1 = h5py.File(file_name_1, 'r')

        data = h5f_handler_0.get('data')
        label_1 = h5f_handler_1.get('data')     # with 0.2 GT
        label_2 = h5f_handler_0.get('label')        # GT
        print ('loading data from:', file_name_0, file_name_1)
        print ('data shape is:', data.shape)
        print ('label_1 shape is:', label_1.shape)
        print ('label_2 shape is:', label_2.shape)

        self.data = data
        self.label_1 = label_1
        self.label_2 = label_2

    def __getitem__(self, index):          
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.label_1[index,:,:,:]).float(), \
            torch.from_numpy(self.label_2[index,:,:,:]).float()

    def __len__(self):
        return self.data.shape[0]
