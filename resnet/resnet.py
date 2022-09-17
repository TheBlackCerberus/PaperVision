import torch 
import numpy as np 
import torch.nn as nn


class BasicResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super().__init__()
        self.expansion = 1
        
        self.conv3x3_1  = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv3x3_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x) -> torch.Tensor:

        identity = x

        out = self.conv3x3_1(x)
        out = self.batch_norm_1(out)
        out = self.relu_1(out)
        out = self.conv3x3_1(out)
        out = self.batch_norm_2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # f(x) + x 
        # during the first pass you will get 
        # f(x) + identity 
        # f(x) = 0, because weights and biases are 0 during first pass.
        # --> so the network will learn the f(identity)
        # --> this preserves the information and helps the network with optimatization
        # --> the rest is learned and added in another passes (Residuals)
        # --> the learnt "out" will be always added with identity --> to be able to perform the operation 
        # the convolutions needs to preserve the format they have 

        out = out + identity
        out = self.relu_1(out)
        return out 

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super().__init__()
        self.expansion = 4

        self.conv1x1_1  = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)
        self.conv3x3_1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)
        self.conv1x1_2  = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*self.expansion,kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm_3 = nn.BatchNorm2d(out_channels*self.expansion) 
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x) -> torch.Tensor:

        identity = x

        out = self.conv1x1_1(x)
        out = self.batch_norm_1(out)
        out = self.relu(out)

        out = self.conv3x3_1(out)
        out = self.batch_norm_2(out)
        out = self.relu(out)

        out = self.conv1x1_2(out)
        out = self.batch_norm_3(out)


        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out 


class ResNet(nn.Module):
    def __init__(self, BasicResidualBlock, layer_list, num_classes, num_channels=3):
        super().__init__()
        self.in_channels = 64
    
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)


        self.layer1 = self._make_layer(BasicResidualBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(BasicResidualBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(BasicResidualBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(BasicResidualBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*BasicResidualBlock.expansion, num_classes)


    def _make_layer(self, BasicResidualBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*BasicResidualBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*BasicResidualBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*BasicResidualBlock.expansion)
            )
            
        layers.append(BasicResidualBlock(self.in_channels, planes, downsample=ii_downsample, stride=stride))
        self.in_channels = planes*BasicResidualBlock.expansion
        
        for i in range(blocks-1):
            layers.append(BasicResidualBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.max_pool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out


def ResNet50(num_classes, channels=3):
    return ResNet(BottleNeck, [3,4,6,3], num_classes, channels)
    
def ResNet101(num_classes, channels=3):
    return ResNet(BottleNeck, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(BottleNeck, [3,8,36,3], num_classes, channels)


