import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABCMeta, abstractmethod
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.utils.data import Dataset
import time
import pickle as pkl
from torch import nn
import torch.optim as optim
import torch.nn.init
import math

device = torch.device('cuda:0')


class UnOptimizedNoiseLayer(nn.Module):
    def __init__(self):
        super(UnOptimizedNoiseLayer, self).__init__()

    def forward(self, input):
        if self.training:
            return input + torch.randn(input.shape, device=device)
        else:
            return input


class RandomLinearFunction(Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        layer_input, layer_weight, layer_bias, layer_sigma = args
        layer_noise = torch.randn(size=[len(layer_input), len(layer_weight[0])])
        layer_noise = layer_noise.to(device)
        # print(layer_sigma.device)
        layer_sigma = torch.abs(layer_sigma)
        layer_noise *= layer_sigma
        ctx.save_for_backward(layer_input, layer_weight, layer_bias, layer_sigma, layer_noise)
        output = layer_input.mm(layer_weight) + layer_bias + layer_noise
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        # print(grad_output.shape)
        layer_input, layer_weight, layer_bias, layer_sigma, layer_noise = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_sigma = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(layer_weight.t())
        if ctx.needs_input_grad[1]:
            grad_weight = layer_input.t().mm(grad_output)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        if ctx.needs_input_grad[3]:
            grad_sigma = grad_output * layer_noise / layer_sigma
        return grad_input, grad_weight, grad_bias, grad_sigma


def randomlinearfunction(layer_input, layer_weight, layer_bias, layer_sigma):
    return RandomLinearFunction()(layer_input, layer_weight, layer_bias, layer_sigma)


class RandomLinearLayer(nn.Module):
    def __init__(self, input_features, output_features):
        super(RandomLinearLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = torch.Tensor(input_features, output_features)
        self.bias = torch.Tensor(output_features)
        self.sigma = torch.Tensor(output_features)
        # self.weight.data.random_()
        # self.bias.data.uniform_(-0.1, 0.1)
        # self.sigma.data.uniform_(1.0, 2.0)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        nn.init.normal_(self.bias)
        nn.init.normal_(self.sigma)
        self.weight = self.weight.to(device)
        self.bias = self.bias.to(device)
        self.sigma = self.sigma.to(device)
        self.weight = nn.Parameter(self.weight)
        self.bias = nn.Parameter(self.bias)
        self.sigma = nn.Parameter(self.sigma)

    def forward(self, input, pure=False):
        if self.training:
            return RandomLinearFunction.apply(input, self.weight, self.bias, self.sigma)
        if not pure:
            return RandomLinearFunction.apply(input, self.weight, self.bias, self.sigma)
        else:
            return input.mm(self.weight) + self.bias


class ConvNoiseFunction(Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        # layer_input, layer_sigma, layer_u = args
        layer_input, layer_sigma, layer_u = args
        layer_noise = torch.randn(size=layer_input.shape, device=device)

        layer_sigma = torch.abs(layer_sigma)

        output = layer_input + layer_noise * layer_sigma + layer_u
        # output = layer_input + layer_noise * layer_sigma
        ctx.save_for_backward(layer_input, layer_sigma, layer_noise * layer_sigma, layer_u)
        # ctx.save_for_backward(layer_input, layer_sigma, layer_noise * layer_sigma)
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        # print(grad_output.shape)
        layer_input, layer_sigma, layer_noise, layer_u = ctx.saved_tensors
        # layer_input, layer_sigma, layer_noise = ctx.saved_tensors
        grad_input = grad_sigma = grad_u = None
        # print(grad_output.shape)
        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        if ctx.needs_input_grad[1]:
            grad_sigma = grad_output * layer_noise / layer_sigma
        if ctx.needs_input_grad[2]:
            grad_u = grad_output
        return grad_input, grad_sigma, grad_u


def convnoisefunction(layer_input, layer_sigma, layer_u):
    return ConvNoiseFunction(layer_input, layer_sigma, layer_u)


class OptimizedNoiseLayer(nn.Module):
    def __init__(self, input_features):
        super(OptimizedNoiseLayer, self).__init__()
        self.sigma = torch.Tensor(*input_features)
        self.u = torch.Tensor(*input_features)
        self.sigma.requires_grad = True
        self.u.requires_grad = True
        # print(self.sigma.shape)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.sigma)
        nn.init.xavier_normal_(self.u)
        self.sigma = self.sigma.to(device)
        self.sigma = nn.Parameter(self.sigma, requires_grad=True)
        self.u = self.u.to(device)
        self.u = nn.Parameter(self.u, requires_grad=True)
        # print(self.sigma.shape)

    def forward(self, input, pure=False):
        if self.training:
            # self.u.data = torch.clamp(self.u.data, -1, 1)
            return ConvNoiseFunction.apply(input, self.sigma, self.u)
            # return ConvNoiseFunction.apply(input, self.sigma)
        else:
            if pure:
                return input
            else:
                return ConvNoiseFunction.apply(input, self.sigma, self.u)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        # 残差块的第一个卷积
        # 通道数变换in->out，每一层（除第一层外）的第一个block
        # 图片尺寸变换：stride=2时，w-3+2 / 2 + 1 = w/2，w/2 * w/2
        # stride=1时尺寸不变，w-3+2 / 1 + 1 = w
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # 残差块的第二个卷积
        # 通道数、图片尺寸均不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差块的shortcut
        # 如果残差块的输入输出通道数不同，则需要变换通道数及图片尺寸，以和residual部分相加
        # 输出：通道数*2 图片尺寸/2
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # 通道数相同，无需做变换，在forward中identity = x
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


'''
定义网络结构
'''


class ResNet34(nn.Module):
    def __init__(self, block):
        super(ResNet34, self).__init__()

        # 初始卷积层核池化层
        self.first = nn.Sequential(
            # 卷基层1：7*7kernel，2stride，3padding，outmap：32-7+2*3 / 2 + 1，16*16
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 最大池化，3*3kernel，1stride（32的原始输入图片较小，不再缩小尺寸），1padding，
            # outmap：16-3+2*1 / 1 + 1，16*16
            nn.MaxPool2d(3, 1, 1)
        )

        # 第一层，通道数不变
        self.layer1 = self.make_layer(block, 64, 64, 3, 1)

        # 第2、3、4层，通道数*2，图片尺寸/2
        self.layer2 = self.make_layer(block, 64, 128, 4, 2)  # 输出8*8
        self.layer3 = self.make_layer(block, 128, 256, 6, 2)  # 输出4*4
        self.layer4 = self.make_layer(block, 256, 512, 3, 2)  # 输出2*2

        self.avg_pool = nn.AvgPool2d(2)  # 输出512*1
        self.fc = nn.Linear(512, 10)

    def make_layer(self, block, in_channels, out_channels, block_num, stride):
        layers = []

        # 每一层的第一个block，通道数可能不同
        layers.append(block(in_channels, out_channels, stride))

        # 每一层的其他block，通道数不变，图片尺寸不变
        for i in range(block_num - 1):
            layers.append(block(out_channels, out_channels, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)

        # x.size()[0]: batch size
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x


def getResnet34():
    return ResNet34(ResBlock)


class ResBlockPlus(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockPlus, self).__init__()

        # 残差块的第一个卷积
        # 通道数变换in->out，每一层（除第一层外）的第一个block
        # 图片尺寸变换：stride=2时，w-3+2 / 2 + 1 = w/2，w/2 * w/2
        # stride=1时尺寸不变，w-3+2 / 1 + 1 = w
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 残差块的第二个卷积
        # 通道数、图片尺寸均不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差块的shortcut
        # 如果残差块的输入输出通道数不同，则需要变换通道数及图片尺寸，以和residual部分相加
        # 输出：通道数*2 图片尺寸/2
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # 通道数相同，无需做变换，在forward中identity = x
            self.downsample = None
        self.noise_layer = UnOptimizedNoiseLayer()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.noise_layer(out)
        out = self.relu(out)

        return out


class ResNetPlus34(nn.Module):
    def __init__(self, block):
        super(ResNetPlus34, self).__init__()

        # 初始卷积层核池化层
        self.first = nn.Sequential(
            # 卷基层1：7*7kernel，2stride，3padding，outmap：32-7+2*3 / 2 + 1，16*16
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            UnOptimizedNoiseLayer(),
            # 最大池化，3*3kernel，1stride（32的原始输入图片较小，不再缩小尺寸），1padding，
            # outmap：16-3+2*1 / 1 + 1，16*16
            nn.MaxPool2d(3, 1, 1)
        )

        # 第一层，通道数不变
        self.layer1 = self.make_layer(block, 64, 64, 3, 1)
        self.relu = nn.ReLU()
        # 第2、3、4层，通道数*2，图片尺寸/2
        self.layer2 = self.make_layer(block, 64, 128, 4, 2)  # 输出8*8
        self.layer3 = self.make_layer(block, 128, 256, 6, 2)  # 输出4*4
        self.layer4 = self.make_layer(block, 256, 512, 3, 2)  # 输出2*2
        self.avg_pool = nn.AvgPool2d(2)  # 输出512*1
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.noise_layer = UnOptimizedNoiseLayer()

    def make_layer(self, block, in_channels, out_channels, block_num, stride):
        layers = []

        # 每一层的第一个block，通道数可能不同
        layers.append(block(in_channels, out_channels, stride))

        # 每一层的其他block，通道数不变，图片尺寸不变
        for i in range(block_num - 1):
            layers.append(block(out_channels, out_channels, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)

        # x.size()[0]: batch size
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.noise_layer(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.noise_layer(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.noise_layer(x)

        return x


def getResnetPlus34():
    return ResNetPlus34(ResBlock)


class ResBlockN(nn.Module):
    def __init__(self, in_channels, out_channels, output_shape, stride=1):
        super(ResBlockN, self).__init__()

        # 残差块的第一个卷积
        # 通道数变换in->out，每一层（除第一层外）的第一个block
        # 图片尺寸变换：stride=2时，w-3+2 / 2 + 1 = w/2，w/2 * w/2
        # stride=1时尺寸不变，w-3+2 / 1 + 1 = w
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.noise_layer = OptimizedNoiseLayer(output_shape)
        # 残差块的第二个卷积
        # 通道数、图片尺寸均不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差块的shortcut
        # 如果残差块的输入输出通道数不同，则需要变换通道数及图片尺寸，以和residual部分相加
        # 输出：通道数*2 图片尺寸/2
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # 通道数相同，无需做变换，在forward中identity = x
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.noise_layer(out)
        out = self.relu(out)

        return out


class ResNetN34(nn.Module):
    def __init__(self, block):
        super(ResNetN34, self).__init__()

        # 初始卷积层核池化层
        self.first = nn.Sequential(
            # 卷基层1：7*7kernel，2stride，3padding，outmap：32-7+2*3 / 2 + 1，16*16
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            OptimizedNoiseLayer((64, 16, 16)),
            nn.ReLU(),
            # 最大池化，3*3kernel，1stride（32的原始输入图片较小，不再缩小尺寸），1padding，
            # outmap：16-3+2*1 / 1 + 1，16*16
            nn.MaxPool2d(3, 1, 1)
        )

        # 第一层，通道数不变
        self.layer1 = self.make_layer(block, 64, 64, 3, 1, (64, 16, 16))

        # 第2、3、4层，通道数*2，图片尺寸/2
        self.layer2 = self.make_layer(block, 64, 128, 4, 2, (128, 8, 8))  # 输出8*8
        self.layer3 = self.make_layer(block, 128, 256, 6, 2, (256, 4, 4))  # 输出4*4
        self.layer4 = self.make_layer(block, 256, 512, 3, 2, (512, 2, 2))  # 输出2*2
        self.avg_pool = nn.AvgPool2d(2)  # 输出512*1
        self.fc1 = RandomLinearLayer(512, 256)
        self.fc2 = RandomLinearLayer(256, 128)
        self.fc3 = RandomLinearLayer(128, 10)
        self.relu = nn.ReLU()

    def make_layer(self, block, in_channels, out_channels, block_num, stride, output_shape):
        layers = []

        # 每一层的第一个block，通道数可能不同
        layers.append(block(in_channels, out_channels, output_shape, stride))

        # 每一层的其他block，通道数不变，图片尺寸不变
        for i in range(block_num - 1):
            layers.append(block(out_channels, out_channels, output_shape, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)

        # x.size()[0]: batch size
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x


def getResnetN34():
    return ResNetN34(ResBlockN)
