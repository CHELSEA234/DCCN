import torch
import torch.nn as nn
from math import sqrt

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)   # output value will be overwritten
        
    def forward(self, x):
        return self.relu(self.conv(x))
        
class Net(nn.Module):
    def __init__(self, layers):
        # claim 3 types layer here, and connect them in the forward
        super(Net, self).__init__()

        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.residual_layer_1 = self.make_layer(Conv_ReLU_Block, int(layers/2))
        self.residual_layer_2 = self.make_layer(Conv_ReLU_Block, int(layers/2))
        
        self.output_1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.output_2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
    
        # this is the kernel initialization
        for m in self.modules():    # this is your model
            if isinstance(m, nn.Conv2d):    # isinstance（object，type）
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                
    def make_layer(self, block, num_of_layer):  # it is like stacked layer together
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))

        out_1 = self.residual_layer_1(out)
        output_1 = self.output_1(out_1)

        out_2 = self.residual_layer_2(out_1)
        output_2 = self.output_2(out_2)

        output_1 = torch.add(output_1,residual)
        output_2 = torch.add(output_2,residual)

        return output_1, output_2

        
