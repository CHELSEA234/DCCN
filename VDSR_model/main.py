import sys
sys.path.append(r'/auto/rcf-proj2/jc/xiaoguo/SISR/Helper')
import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from utils import adjust_learning_rate, save_checkpoint
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vdsr_progressive import Net
from VDSR_dataset import DatasetFromHdf5
from testModel import test_phase
from tools import image_plot

parser = argparse.ArgumentParser(description="PyTorch VDSR")
parser.add_argument("--batchSize", type=int, default=64, help="Training batch size")
parser.add_argument("--layers", type=int, default=20, help="How many layers does this net have")
parser.add_argument("--nEpochs", type=int, default=100, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.1, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=20, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--display_step", type=int, default=1, help="To display PSNR and SSIM")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 4")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--scale", default=2, type=int, help="upscaling factor")
parser.add_argument('--figDir', default='figure/', type=str, help='path to store result figure')
parser.add_argument('--paraDir', default='para/', type=str, help='path to store result para')

def main():
    global opt, model
    opt = parser.parse_args()
    print (opt)

    if not os.path.exists(opt.figDir):
        os.makedirs(opt.figDir)

    if not os.path.exists(opt.paraDir):
        os.makedirs(opt.paraDir)

    opt.seed = 1234  # for reproductility
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True
        
    print("===> Loading datasets")
    train_set = DatasetFromHdf5()
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    print("===> Building model")
    intermediate_layer_num = int(opt.layers-2)
    model = Net(layers=intermediate_layer_num)
    criterion = nn.MSELoss(size_average=False)

    print("===> Setting GPU")
    model = torch.nn.DataParallel(model).cuda()
    criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    
    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))  
            
    print("===> Setting Optimizer")
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
            
    print("===> Training")
    psnr_value_list = []
    ssim_value_list = []


    f = open('out_1.txt', 'w')
    f.write('it can work Epoch:{}'.format(1))
    f.close()

    for epoch in range(opt.start_epoch, opt.nEpochs + 1): 
        f = open('out_2.txt', 'w')
        f.write('entering epoch')
        f.close()
        lr = adjust_learning_rate(optimizer, epoch-1, opt)
        
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
     
        print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
        model.train()    

        for iteration, batch in enumerate(training_data_loader, 1):
            f = open('out_3.txt', 'w')
            f.write('entering iteration')
            f.close()

            input, label_1, label_2 = Variable(batch[0]), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False)
            input = input.cuda()
            label_1 = label_1.cuda()
            label_2 = label_2.cuda()
            
            output_1, output_2 = model(input)

            loss_1 = criterion(output_1, label_1)
            loss_2 = criterion(output_2, label_2)
            loss = torch.add(loss_1, loss_2)

            optimizer.zero_grad()
            loss_1.backward(retain_variables=True)
            loss_2.backward()
            nn.utils.clip_grad_norm(model.parameters(),opt.clip) 
            optimizer.step()   
        
            # if iteration%100 == 0:
            if iteration%10 == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.10f}, loss_1: {:.10f}, loss_2: {:.10f}".format(epoch, iteration, \
                    len(training_data_loader), loss.data[0], loss_1.data[0], loss_2.data[0]))

                # print >> f, ("===> Epoch[{}]({}/{}): Loss: {:.10f}, loss_1: {:.10f}, loss_2: {:.10f}".format(epoch, iteration, \
                #     len(training_data_loader), loss.data[0], loss_1.data[0], loss_2.data[0]))
                f = open('out_5.txt', 'w')
                f.write("===> Epoch[{}]({}/{}): Loss: {:.10f}, loss_1: {:.10f}, loss_2: {:.10f}".format(epoch, iteration, \
                    len(training_data_loader), loss.data[0], loss_1.data[0], loss_2.data[0]))
                f.write("whatever")
                f.close()

        save_checkpoint(model, epoch, iteration, opt)
        psnr_value, ssim_value = test_phase(model, scale=opt.scale, dataset='Set5')
        print ('at epoch', epoch, 'psnr_value is', psnr_value, 'ssim_value is', ssim_value)
        psnr_value_list.append(psnr_value)
        ssim_value_list.append(ssim_value)
        image_plot(psnr_value_list, ssim_value_list, opt.figDir, index = epoch, iteration=iteration)

    # f.close()

if __name__ == "__main__":
    main()
