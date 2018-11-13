# Implementation of project DCCN, densely-connected-cycle-network

*some codes are being organized*.

### Dependencies:

* Python3.6 or 3.5
* Pytorch-gpu 0.1.12
* Matlab R2016b

### Files:

* **VDSR_dataMaking** contains codes for preprocessing raw image dataset into the form that can be used by VDSR; This is the same for **Lap_dataMaking**. During the preprocessing, **separateData** will be used, too.

* **VDSR_model** is the implementation of VDSR, which I have obtain from [here](https://github.com/twtygqyy/pytorch-vdsr).

* **model_test** contains 1)codes for the image augmentation, interpolation, conversion from RGB to crcb type and computation for PSNR and SSIM, 2) pertained models that is in **pretrained**, 3) the test dataset for the evaluation.

* The training model is being organized now.
