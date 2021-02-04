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


def LoadMNIST(root, transform, batch_size, download=True):
    train_dataset = datasets.MNIST(root=root, train=True, transform=transform, download=download)
    test_dataset = datasets.MNIST(root=root, train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

class DenseReLU(nn.Module):
    def __init__(self):
        super(DenseReLU, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DenseSigmoid(nn.Module):
    def __init__(self):
        super(DenseSigmoid, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = nn.Sigmoid()(self.fc1(x))
        x = nn.Sigmoid()(self.fc2(x))
        x = self.fc3(x)
        return x


class UnOptimizedNoiseLayer(nn.Module):
    def __init__(self):
        super(UnOptimizedNoiseLayer, self).__init__()

    def forward(self, input):
        if self.training:
            return input + torch.randn(input.shape, device=device)
        else:
            return input


class DensePlusReLU(nn.Module):
    def __init__(self):
        super(DensePlusReLU, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(28 * 28, 100), UnOptimizedNoiseLayer())
        self.fc2 = nn.Sequential(nn.Linear(100, 50), UnOptimizedNoiseLayer())
        self.fc3 = nn.Sequential(nn.Linear(50, 10), UnOptimizedNoiseLayer())

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DensePlusSigmoid(nn.Module):
    def __init__(self):
        super(DensePlusSigmoid, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(28 * 28, 100), UnOptimizedNoiseLayer())
        self.fc2 = nn.Sequential(nn.Linear(100, 50), UnOptimizedNoiseLayer())
        self.fc3 = nn.Sequential(nn.Linear(50, 10), UnOptimizedNoiseLayer())

    def forward(self, x):
        x = nn.Sigmoid()(self.fc1(x))
        x = nn.Sigmoid()(self.fc2(x))
        x = self.fc3(x)
        return x




class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(1152, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNFCplus(nn.Module):
    """
    全连接加无优化噪声
    """

    def __init__(self):
        super(CNNFCplus, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(1152, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc1(x)
        if self.training:
            x += torch.randn(x.shape).to(device)
        x = F.relu(x)
        x = self.fc2(x)
        if self.training:
            x += torch.randn(x.shape).to(device)
        return x


class CNNAplus(nn.Module):
    """
    卷积和全连接均加无优化噪声
    """

    def __init__(self):
        super(CNNAplus, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1),
            UnOptimizedNoiseLayer(),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            UnOptimizedNoiseLayer(),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(1152, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc1(x)
        if self.training:
            x += torch.randn(x.shape).to(device)
        x = F.relu(x)
        x = self.fc2(x)
        if self.training:
            x += torch.randn(x.shape).to(device)
        return x


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


class DenseNoiseReLU(nn.Module):
    def __init__(self):
        super(DenseNoiseReLU, self).__init__()
        self.fc1 = nn.Sequential(RandomLinearLayer(28 * 28, 100))
        self.fc2 = nn.Sequential(RandomLinearLayer(100, 50))
        self.fc3 = nn.Sequential(RandomLinearLayer(50, 10))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DenseNoiseSigmoid(nn.Module):
    def __init__(self):
        super(DenseNoiseSigmoid, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(28 * 28, 100), UnOptimizedNoiseLayer())
        self.fc2 = nn.Sequential(nn.Linear(100, 50), UnOptimizedNoiseLayer())
        self.fc3 = nn.Sequential(nn.Linear(50, 10), UnOptimizedNoiseLayer())

    def forward(self, x):
        x = nn.Sigmoid()(self.fc1(x))
        x = nn.Sigmoid()(self.fc2(x))
        x = self.fc3(x)
        return x

class CNNFCN(nn.Module):
    def __init__(self):
        super(CNNFCN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc1 = RandomLinearLayer(1152, 128)
        self.fc2 = RandomLinearLayer(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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


class CNNAN(nn.Module):
    def __init__(self):
        super(CNNAN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1),
            OptimizedNoiseLayer((32, 28, 28)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            OptimizedNoiseLayer((64, 14, 14)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc1 = RandomLinearLayer(1152, 128)
        self.fc2 = RandomLinearLayer(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
