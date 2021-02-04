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

    def forward(self, input, pure=True):
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

    def forward(self, input, pure=True):
        if self.training:
            # self.u.data = torch.clamp(self.u.data, -1, 1)
            return ConvNoiseFunction.apply(input, self.sigma, self.u)
            # return ConvNoiseFunction.apply(input, self.sigma)
        else:
            if pure:
                return input
            else:
                return ConvNoiseFunction.apply(input, self.sigma, self.u)


class ResidualBlock(nn.Module):  # 继承nn.Module
    def __init__(self, inchannel, outchannel, stride=1):  # __init()中必须自己定义可学习的参数
        super(ResidualBlock, self).__init__()  # 调用nn.Module的构造函数
        self.left = nn.Sequential(  # 左边，指残差块中按顺序执行的普通卷积网络
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),  # 最常用于卷积网络中(防止梯度消失或爆炸)
            nn.LeakyReLU(),  # implace=True是把输出直接覆盖到输入中，节省内存
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:  # 只有步长为1并且输入通道和输出通道相等特征图大小才会一样，如果不一样，需要在合并之前进行统一
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):  # 实现前向传播过程
        out = self.left(x)  # 先执行普通卷积神经网络
        out += self.shortcut(x)  # 再加上原始x数据
        out = F.leaky_relu(out)
        return out


"""整个卷积网络，包含若干个残差块"""


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),  # 设置参数为卷积的输出通道数
            nn.LeakyReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)  # 一个残差单元，每个单元中国包含2个残差块
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )  # 全连接层(1,512)-->(1,10)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (
                num_blocks - 1)  # 将该单元中所有残差块的步数做成一个一个向量，第一个残差块的步数由传入参数指定，后边num_blocks-1个残差块的步数全部为1，第一个单元为[1,1]，后边三个单元为[2,1]
        layers = []
        for stride in strides:  # 对每个残差块的步数进行迭代
            layers.append(block(self.inchannel, channels, stride))  # 执行每一个残差块，定义向量存储每个残差块的输出值
            self.inchannel = channels
        return nn.Sequential(*layers)  # 如果*加在了实参上，代表的是将向量拆成一个一个的元素

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)  # 平均池化，4*4的局部特征取平均值，最后欸(512,1,1)
        out = out.view(out.size(0), -1)  # 转换为(1,512)的格式
        out = self.fc(out)
        return out

    def predict(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)  # 平均池化，4*4的局部特征取平均值，最后是(512,1,1)
        out = out.view(out.size(0), -1)  # 转换为(1,512)的格式
        out = self.fc(out)
        return out


def getResNet18():
    return ResNet(ResidualBlock)

class ResidualBlockPlusFC(nn.Module):  # 继承nn.Module
    def __init__(self, inchannel, outchannel, stride=1):  # __init()中必须自己定义可学习的参数
        super(ResidualBlockPlusFC, self).__init__()  # 调用nn.Module的构造函数
        self.left = nn.Sequential(  # 左边，指残差块中按顺序执行的普通卷积网络
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),  # 最常用于卷积网络中(防止梯度消失或爆炸)
            nn.LeakyReLU(inplace=True),  # implace=True是把输出直接覆盖到输入中，节省内存
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:  # 只有步长为1并且输入通道和输出通道相等特征图大小才会一样，如果不一样，需要在合并之前进行统一
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel),
            )

    def forward(self, x):  # 实现前向传播过程
        out = self.left(x)  # 先执行普通卷积神经网络
        out += self.shortcut(x)  # 再加上原始x数据
        out = F.relu(out)
        return out


"""整个卷积网络，包含若干个残差块"""


class ResNetPlus18FC(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNetPlus18FC, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),  # 设置参数为卷积的输出通道数
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)  # 一个残差单元，每个单元中国包含2个残差块
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Sequential(nn.Linear(512, 256),
                                UnOptimizedNoiseLayer(),
                                nn.LeakyReLU(),
                                nn.Linear(256, 128),
                                UnOptimizedNoiseLayer(),
                                nn.LeakyReLU(),
                                nn.Linear(128, num_classes),
                                )  # 全连接层(1,512)-->(1,10)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (
                num_blocks - 1)  # 将该单元中所有残差块的步数做成一个一个向量，第一个残差块的步数由传入参数指定，后边num_blocks-1个残差块的步数全部为1，第一个单元为[1,1]，后边三个单元为[2,1]
        layers = []
        for stride in strides:  # 对每个残差块的步数进行迭代
            layers.append(block(self.inchannel, channels, stride))  # 执行每一个残差块，定义向量存储每个残差块的输出值
            self.inchannel = channels
        return nn.Sequential(*layers)  # 如果*加在了实参上，代表的是将向量拆成一个一个的元素

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)  # 平均池化，4*4的局部特征取平均值，最后欸(512,1,1)
        out = out.view(out.size(0), -1)  # 转换为(1,512)的格式
        out = self.fc(out)
        return out

def getResNet18PlusFC():
    return ResNetPlus18FC(ResidualBlockPlusFC)


class ResidualBlockPlusA(nn.Module):  # 继承nn.Module
    def __init__(self, inchannel, outchannel, stride=1):  # __init()中必须自己定义可学习的参数
        super(ResidualBlockPlusA, self).__init__()  # 调用nn.Module的构造函数
        self.left = nn.Sequential(  # 左边，指残差块中按顺序执行的普通卷积网络
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),  # 最常用于卷积网络中(防止梯度消失或爆炸)
            nn.LeakyReLU(inplace=True),  # implace=True是把输出直接覆盖到输入中，节省内存
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:  # 只有步长为1并且输入通道和输出通道相等特征图大小才会一样，如果不一样，需要在合并之前进行统一
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel),
            )

    def forward(self, x):  # 实现前向传播过程
        out = self.left(x)  # 先执行普通卷积神经网络
        out += self.shortcut(x)  # 再加上原始x数据
        if self.training:
            out += torch.randn(out.shape, device=device)
        out = F.relu(out)
        return out


"""整个卷积网络，包含若干个残差块"""


class ResNetPlus18A(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNetPlus18A, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),  # 设置参数为卷积的输出通道数
            UnOptimizedNoiseLayer(),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)  # 一个残差单元，每个单元中国包含2个残差块
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Sequential(nn.Linear(512, 256),
                                UnOptimizedNoiseLayer(),
                                nn.LeakyReLU(),
                                nn.Linear(256, 128),
                                UnOptimizedNoiseLayer(),
                                nn.LeakyReLU(),
                                nn.Linear(128, num_classes),
                                )  # 全连接层(1,512)-->(1,10)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (
                num_blocks - 1)  # 将该单元中所有残差块的步数做成一个一个向量，第一个残差块的步数由传入参数指定，后边num_blocks-1个残差块的步数全部为1，第一个单元为[1,1]，后边三个单元为[2,1]
        layers = []
        for stride in strides:  # 对每个残差块的步数进行迭代
            layers.append(block(self.inchannel, channels, stride))  # 执行每一个残差块，定义向量存储每个残差块的输出值
            self.inchannel = channels
        return nn.Sequential(*layers)  # 如果*加在了实参上，代表的是将向量拆成一个一个的元素

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)  # 平均池化，4*4的局部特征取平均值，最后欸(512,1,1)
        out = out.view(out.size(0), -1)  # 转换为(1,512)的格式
        out = self.fc(out)
        return out

def getResNet18PlusA():
    return ResNetPlus18A(ResidualBlockPlusA)

class ResidualBlockNFC(nn.Module):  # 继承nn.Module
    def __init__(self, inchannel, outchannel, output_size, stride=1):  # __init()中必须自己定义可学习的参数
        super(ResidualBlockNFC, self).__init__()  # 调用nn.Module的构造函数
        self.left = nn.Sequential(  # 左边，指残差块中按顺序执行的普通卷积网络
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),  # 最常用于卷积网络中(防止梯度消失或爆炸)
            # ConvNoiseLayer(output_size),
            nn.LeakyReLU(),  # implace=True是把输出直接覆盖到输入中，节省内存
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            # ConvNoiseLayer(output_size),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:  # 只有步长为1并且输入通道和输出通道相等特征图大小才会一样，如果不一样，需要在合并之前进行统一
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel),
                # ConvNoiseLayer(output_size),
            )
        self.block_noise_layer = OptimizedNoiseLayer(output_size)

    def forward(self, x):  # 实现前向传播过程
        # x = self.pre_block_noise_layer(x)
        out = self.left(x)  # 先执行普通卷积神经网络
        out = out + self.shortcut(x)  # 再加上原始x数据
        out = F.leaky_relu(out)
        return out


"""整个卷积网络，包含若干个残差块"""


class ResNetNFC(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNetNFC, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),  # 设置参数为卷积的输出通道数
            nn.LeakyReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, (64, 32, 32), stride=1)  # 一个残差单元，每个单元中国包含2个残差块
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, (128, 16, 16), stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, (256, 8, 8), stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, (512, 4, 4), stride=2)
        self.fc = nn.Sequential(RandomLinearLayer(512, 256),
                                nn.LeakyReLU(),
                                RandomLinearLayer(256, 128),
                                nn.LeakyReLU(),
                                RandomLinearLayer(128, num_classes)
                                )  # 全连接层(1,512)-->(1,10)

    def make_layer(self, block, channels, num_blocks, output_size, stride):
        strides = [stride] + [1] * (
                num_blocks - 1)  # 将该单元中所有残差块的步数做成一个一个向量，第一个残差块的步数由传入参数指定，后边num_blocks-1个残差块的步数全部为1，第一个单元为[1,1]，后边三个单元为[2,1]
        layers = []
        for stride in strides:  # 对每个残差块的步数进行迭代
            layers.append(block(self.inchannel, channels, output_size, stride))  # 执行每一个残差块，定义向量存储每个残差块的输出值
            self.inchannel = channels
        return nn.Sequential(*layers)  # 如果*加在了实参上，代表的是将向量拆成一个一个的元素

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)  # 平均池化，4*4的局部特征取平均值，最后欸(512,1,1)
        out = out.view(out.size(0), -1)  # 转换为(1,512)的格式
        out = self.fc(out)
        return out


def getResNet18NFC():
    return ResNetNFC(ResidualBlockNFC)



class ResidualBlockNA(nn.Module):  # 继承nn.Module
    def __init__(self, inchannel, outchannel, output_size, stride=1):  # __init()中必须自己定义可学习的参数
        super(ResidualBlockNA, self).__init__()  # 调用nn.Module的构造函数
        self.left = nn.Sequential(  # 左边，指残差块中按顺序执行的普通卷积网络
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),  # 最常用于卷积网络中(防止梯度消失或爆炸)
            # ConvNoiseLayer(output_size),
            nn.LeakyReLU(),  # implace=True是把输出直接覆盖到输入中，节省内存
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            # ConvNoiseLayer(output_size),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:  # 只有步长为1并且输入通道和输出通道相等特征图大小才会一样，如果不一样，需要在合并之前进行统一
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel),
                # ConvNoiseLayer(output_size),
            )
        self.block_noise_layer = OptimizedNoiseLayer(output_size)

    def forward(self, x):  # 实现前向传播过程
        # x = self.pre_block_noise_layer(x)
        out = self.left(x)  # 先执行普通卷积神经网络
        out = out + self.shortcut(x)  # 再加上原始x数据
        out = self.block_noise_layer(out)
        out = F.leaky_relu(out)
        return out


"""整个卷积网络，包含若干个残差块"""


class ResNetNA(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNetNA, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),  # 设置参数为卷积的输出通道数
            OptimizedNoiseLayer((64, 32, 32)),
            nn.LeakyReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, (64, 32, 32), stride=1)  # 一个残差单元，每个单元中国包含2个残差块
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, (128, 16, 16), stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, (256, 8, 8), stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, (512, 4, 4), stride=2)
        self.fc = nn.Sequential(RandomLinearLayer(512, 256),
                                nn.LeakyReLU(),
                                RandomLinearLayer(256, 128),
                                nn.LeakyReLU(),
                                RandomLinearLayer(128, num_classes)
                                )  # 全连接层(1,512)-->(1,10)

    def make_layer(self, block, channels, num_blocks, output_size, stride):
        strides = [stride] + [1] * (
                num_blocks - 1)  # 将该单元中所有残差块的步数做成一个一个向量，第一个残差块的步数由传入参数指定，后边num_blocks-1个残差块的步数全部为1，第一个单元为[1,1]，后边三个单元为[2,1]
        layers = []
        for stride in strides:  # 对每个残差块的步数进行迭代
            layers.append(block(self.inchannel, channels, output_size, stride))  # 执行每一个残差块，定义向量存储每个残差块的输出值
            self.inchannel = channels
        return nn.Sequential(*layers)  # 如果*加在了实参上，代表的是将向量拆成一个一个的元素

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)  # 平均池化，4*4的局部特征取平均值，最后欸(512,1,1)
        out = out.view(out.size(0), -1)  # 转换为(1,512)的格式
        out = self.fc(out)
        return out


def getResNet18NA():
    return ResNetNA(ResidualBlockNA)