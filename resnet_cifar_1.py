'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['resnet18_c1']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bIndex=0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.conv1.retain_grad()
        self.actLayer = {}
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        #self.conv2.retain_grad()
        self.bn2 = nn.BatchNorm2d(planes)
        self.bIndex = bIndex

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        x1 = x
        out = F.relu(self.bn1(self.conv1(x1)))
        self.actLayer[self.bIndex] = out
        out = self.bn2(self.conv2(out))
        self.actLayer[self.bIndex+1] = out
        out += self.shortcut(x)
        out = F.relu(out)
        return out, self.actLayer


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, layers):
        super(ResBlock, self).__init__()

        self.layers = layers
        #self.lr_con = lr_con
        self.actLayer = {}

    def forward(self, x):
        numLayers = len(self.layers)
        out = x
        for i in range(0,numLayers):
            out, act = self.layers[i](out)
            self.actLayer.update(act)
        return out, self.actLayer


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bIndex = 0
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, bIndex=0)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, bIndex=4)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, bIndex=8)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, bIndex=12)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.actLayer = {}

    def _make_layer(self, block, planes, num_blocks, stride, bIndex):
        strides = [stride] + [1]*(num_blocks-1)
        layers = nn.ModuleList([]) #[]
        bInd = bIndex
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bInd))
            bInd = bInd + 1
            self.in_planes = planes * block.expansion
        return ResBlock(layers) #nn.Sequential(*layers)

    def forward(self, x):
        
        out = F.relu(self.bn1(self.conv1(x)))
        out, actl = self.layer1(out)
        self.actLayer.update(actl)
        out, actl = self.layer2(out)
        self.actLayer.update(actl)
        out, actl = self.layer3(out)
        self.actLayer.update(actl)
        out, actl = self.layer4(out)
        self.actLayer.update(actl)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, self.actLayer


def resnet18_c1():
    return ResNet(BasicBlock, [2, 2, 2, 2])




