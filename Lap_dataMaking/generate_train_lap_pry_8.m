clear;
close all;
% folder = '/Users/guoxiao/Desktop/research/data/Train_291';
folder = '/auto/rcf-proj2/jc/xiaoguo/SISR/train_dataset/Train_291';
% folder = '/Users/guoxiao/Desktop/research/data/Set5';

% savepath = '/Users/guoxiao/Desktop/research/data/Train_291/train.h5';
% savepath = '/Users/guoxiao/Desktop/research/data/Set5/train_x8.h5';
savepath = '/auto/rcf-proj2/jc/xiaoguo/SISR/train_dataset/train_x8.h5';

size_label = 128;
scale = 8;
size_input = size_label/scale;      %   16
size_x2 = size_label/4;         %   32
size_x4 = size_label/2;         %   64
stride = 64;
%% downsizing
downsizes = [1,0.7,0.5];

data = zeros(size_input, size_input, 1, 1);     %   16 x 16
label_x2 = zeros(size_x2, size_x2, 1, 1);       %   32 x 32
label_x4 = zeros(size_x4, size_x4, 1, 1);       %   64 x 64
label_x8 = zeros(size_label, size_label, 1, 1);     %   128 x 128
count = 0;
margain = 0;

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];

length(filepaths)

for i = 1 : length(filepaths)
    for downsize = 1 : length(downsizes)
        image = imread(fullfile(folder,filepaths(i).name));
        image = imresize(image,downsizes(downsize),'bicubic');

        if size(image,3)==3
            image = rgb2ycbcr(image);
            image = im2double(image(:, :, 1));
            im_label = modcrop(image, scale);
            [hei,wid] = size(im_label);

            filepaths(i).name
            for x = 1 + margain : stride : hei-size_label+1 - margain
                for y = 1 + margain :stride : wid-size_label+1 - margain
                    subim_label = im_label(x : x+size_label-1, y : y+size_label-1);
                    subim_label_x4 = imresize(subim_label, 1/scale*4,'bicubic');
                    subim_label_x2 = imresize(subim_label, 1/scale*2,'bicubic');
                    subim_input = imresize(subim_label, 1/scale,'bicubic');

                    count=count+1;
                    data(:, :, 1, count) = subim_input;
                    label_x2(:, :, 1, count) = subim_label_x2;
                    label_x4(:, :, 1, count) = subim_label_x4;
                    label_x8(:, :, 1, count) = subim_label;
                end
            end
        end
    end
end

order = randperm(count);
data = data(:, :, 1, order);
label_x8 = label_x8(:, :, 1, order);        %   128 x 128
label_x4 = label_x4(:, :, 1, order);        %   64 x 64
label_x2 = label_x2(:, :, 1, order);        %   32 x 32

%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    batchno
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,1,last_read+1:last_read+chunksz); 
    batchlabs_x2 = label_x2(:,:,1,last_read+1:last_read+chunksz);   % 32x 32
    batchlabs_x4 = label_x4(:,:,1,last_read+1:last_read+chunksz);   % 64x 64
    batchlabs = label_x8(:,:,1,last_read+1:last_read+chunksz);  % 128x 128
    startloc = struct('dat',[1,1,1,totalct+1], 'lab_x2', [1,1,1,totalct+1], 'lab_x4', [1,1,1,totalct+1], 'lab_x8', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5_x8(savepath, batchdata, batchlabs_x2, batchlabs_x4, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);

