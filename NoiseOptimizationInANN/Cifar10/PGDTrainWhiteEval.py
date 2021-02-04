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
import pickle
import Resnet18
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
device = Resnet18.device


def pgd_attack(model, images, labels, eps=8 / 255, alpha=2 / 255, iters=5):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images


def fgsm_attack(model, images, labels, eps=0.1, alpha=0.1, iters=1):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images


def lbfgs_attack(model, images, labels, eps=8 / 255, alpha=0.1, iters=10):
    images = images.to(device)
    # labels = labels.to(device)
    lossfunc = nn.CrossEntropyLoss()

    ori_images = images.data

    xv = nn.Parameter(images, requires_grad=True)
    # xv = nn.Parameter(torch.FloatTensor(x.reshape(1, 28 * 28)), requires_grad=True)
    y_true = Variable(torch.LongTensor(labels.cpu()).to(device), requires_grad=False)
    method = optim.LBFGS([xv], lr=5e-3, max_iter=iters)
    # Classification before Adv
    y_pred = torch.argmax(model(xv), dim=-1)

    # Generate Adversarial Image
    def closure():
        method.zero_grad()
        output = model(xv)
        loss = -lossfunc(output, y_true)
        loss.backward()
        return loss

    method.step(closure)
    # method = optim.LBFGS(list(xv), lr=1e-1)
    # Add perturbation
    # x_grad = torch.sign(x.grad.data)
    x_adversarial = torch.clamp(xv.data, 0, 1)

    return x_adversarial


if __name__ == '__main__':
    import Resnet18

    """resnet18A_model = [Resnet18.getResNet18PlusA, Resnet18.getResNet18NA]

    resnet18A_name = ['Resnet18.getResNet18PlusA', 'Resnet18.getResNet18NA']

    resnet18FC_model = [Resnet18.getResNet18, Resnet18.getResNet18PlusFC, Resnet18.getResNet18NFC]

    resnet18FC_name = ['Resnet18.getResNet18', 'Resnet18.getResNet18PlusFC', 'Resnet18.getResNet18NFC']"""

    resnet18A_model = []

    resnet18A_name = []

    resnet18FC_model = [Resnet18.getResNet18NFC]

    resnet18FC_name = ['Resnet18.getResNet18NFC']


    model_zoo_name = [resnet18FC_name, resnet18A_name]
    noise = ['FGSM', 'LBFGS', 'PGD']
    noise_method = [fgsm_attack, lbfgs_attack, pgd_attack]

    strength = [1, 2, 3, 4, 5]
    model_zoo = [resnet18FC_model, resnet18A_model]
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    batch_size = 128
    epoches = 50
    loss = 0.
    train_dataset = datasets.CIFAR10(root='../../LR/EBP', train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root='../../LR/EBP', train=False, transform=transform_test, download=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    epsilon = 0.1
    info = dict()

    for i in range(len(model_zoo)):
        for j in range(len(model_zoo[i])):
            # print('new model training:{}'.format(model_zoo_name[i][j]))
            dense_base_acc = [0. for _ in range(len(resnet18FC_model))]
            cnn_base_acc = [0. for _ in range(len(resnet18A_model))]
            base_acc = [dense_base_acc, cnn_base_acc]
            adv_acc_hot = [0. for _ in range(len(noise))]
            dense_adv_acc = [adv_acc_hot for _ in range(len(resnet18FC_model))]
            cnn_adv_acc = [adv_acc_hot for _ in range(len(resnet18A_model))]
            adv_acc = [dense_adv_acc, cnn_adv_acc]
            model = model_zoo[i][j]()
            static = torch.load('./PGDmnist_model/{}.pth'.format(model_zoo_name[i][j]), map_location='cpu')
            model.load_state_dict(static)
            model = model.to(device)
            model.eval()
            for noise_kind in range(len(noise)):
                N = 0.
                tmp_base_acc = 0.
                tmp_adv_acc = 0.
                for batch, [inputs, labels] in tqdm(enumerate(test_dataloader)):
                    N += len(inputs)

                    inputs = Variable(torch.FloatTensor(inputs).to(device), requires_grad=True)
                    labels = Variable(torch.LongTensor(labels).to(device), requires_grad=False)
                    base_outputs = model(inputs)
                    base_predict = torch.argmax(base_outputs, dim=-1)
                    tmp_base_acc = np.sum((base_predict.cpu().numpy() == labels.cpu().numpy()))
                    base_acc[i][j] += tmp_base_acc

                    x_adversarial = noise_method[noise_kind](model, inputs, labels)

                    # Classification after optimization
                    y_pred_adversarial = torch.argmax(model(x_adversarial), dim=-1)
                    adv_acc[i][j][noise_kind] += np.sum(y_pred_adversarial.cpu().numpy() == labels.cpu().numpy())
                adv_acc[i][j][noise_kind] /= N
                base_acc[i][j] /= N

            info[model_zoo_name[i][j]] = {
                'base acc': base_acc[i][j],
                'adv acc': adv_acc[i][j]
            }
    for model_name in info.keys():
        print(model_name)
        print(info[model_name])
