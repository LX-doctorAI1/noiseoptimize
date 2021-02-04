import Resnet18
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

device = Resnet18.device


def load_mnist(file, batch_size=100):
    with open(file, "rb") as f:
        adv_data_dict2 = pickle.load(f)
        xs_clean = np.array(adv_data_dict2['xs'])
        y_true_clean = np.array(adv_data_dict2['y_trues'])
        y_preds = adv_data_dict2['y_preds']
        adv_x = np.array(adv_data_dict2['noises'])
        y_preds_adversarial = adv_data_dict2['y_preds_adversarial']
        """
        y_true_clean = np.array(y_true_clean).reshape(-1)
        batch_clean = []
        batch_adv = []
        batch_y = []
        index = np.arange(len(xs_clean))
        batch = [index[i:i + batch_size] for i in range(0, len(index), batch_size)]
        batch_clean = [xs_clean[b] for b in batch]
        batch_adv = [adv_x[b] for b in batch]
        batch_y = [y_true_clean[b] for b in batch]"""
    return xs_clean, adv_x, y_true_clean


if __name__ == '__main__':

    resnet18A_model = [Resnet18.getResNet18PlusA, Resnet18.getResNet18NA]

    resnet18A_name = ['Resnet18.getResNet18PlusA', 'Resnet18.getResNet18NA']

    resnet18FC_model = [Resnet18.getResNet18, Resnet18.getResNet18PlusFC, Resnet18.getResNet18NFC]

    resnet18FC_name = ['Resnet18.getResNet18', 'Resnet18.getResNet18PlusFC', 'Resnet18.getResNet18NFC']

    model_zoo_name = [resnet18FC_name, resnet18A_name]
    noise = ['gaussian', 'impulse', 'glass_blur', 'contrast', 'FGSM', 'LBFGS']

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
            static = torch.load('./mnist_model/{}.pth'.format(model_zoo_name[i][j]), map_location='cpu')
            model.load_state_dict(static)
            model = model.to(device)
            for noise_kind in range(len(noise)):
                if noise[noise_kind] == 'FGSM' or noise[noise_kind] == 'LBFGS':
                    file = "Generate_adversarial_sample_epsilon=" + str(epsilon) + "by_{}.pkl".format(
                        noise[noise_kind])
                    batch_clean, batch_adv, batch_y = load_mnist(file, batch_size=100)
                    N = 0.
                    tmp_base_acc = 0.
                    tmp_adv_acc = 0.
                    for idx in tqdm(range(len(batch_y))):
                        adv_img = batch_adv[idx].reshape(-1, 3, 32, 32)
                        clean_img = batch_clean[idx].reshape(-1, 3, 32, 32)
                        label = batch_y[idx]
                        N += len(label)
                        if i == 1:
                            adv_img = adv_img.reshape(-1, 3, 32, 32)
                            clean_img = clean_img.reshape(-1, 3, 32, 32)
                        adv_img = torch.from_numpy(adv_img).to(device)
                        clean_img = torch.from_numpy(clean_img).to(device)
                        advpred_based_bp = model(adv_img)
                        cleanpred_based_bp = model(clean_img)

                        advpred_based_bp = torch.argmax(advpred_based_bp, dim=-1).cpu().numpy()
                        cleanpred_based_bp = torch.argmax(cleanpred_based_bp, dim=-1).cpu().numpy()

                        tmp_base_acc += np.array(cleanpred_based_bp == label).sum()
                        tmp_adv_acc += np.array(advpred_based_bp == label).sum()

                    tmp_adv_acc /= N
                    tmp_base_acc /= N
                    base_acc[i][j] = tmp_base_acc
                    adv_acc[i][j][noise_kind] = tmp_adv_acc
                    continue
                for s in strength:
                    file = "Generate_adversarial_sample_epsilon=" + str(epsilon) + "by_{}_{}.pkl".format(
                        noise[noise_kind], s)
                    batch_clean, batch_adv, batch_y = load_mnist(file, batch_size=100)
                    N = 0.
                    tmp_base_acc = 0.
                    tmp_adv_acc = 0.
                    for idx in tqdm(range(len(batch_y))):
                        adv_img = batch_adv[idx].reshape(-1, 3, 32, 32)
                        clean_img = batch_clean[idx].reshape(-1, 3, 32, 32)
                        label = batch_y[idx]
                        N += len(label)
                        if i == 1:
                            adv_img = adv_img.reshape(-1, 3, 32, 32)
                            clean_img = clean_img.reshape(-1, 3, 32, 32)
                        adv_img = torch.from_numpy(adv_img).to(device)
                        clean_img = torch.from_numpy(clean_img).to(device)
                        advpred_based_bp = model(adv_img)
                        cleanpred_based_bp = model(clean_img)

                        advpred_based_bp = torch.argmax(advpred_based_bp, dim=-1).cpu().numpy()
                        cleanpred_based_bp = torch.argmax(cleanpred_based_bp, dim=-1).cpu().numpy()
                        tmp_base_acc += np.array(cleanpred_based_bp == label).sum()
                        tmp_adv_acc += np.array(advpred_based_bp == label).sum()
                    tmp_adv_acc /= N
                    tmp_base_acc /= N
                    print(i, j)
                    base_acc[i][j] = tmp_base_acc
                    adv_acc[i][j][noise_kind] += tmp_adv_acc
                adv_acc[i][j][noise_kind] /= len(strength)
                info[model_zoo_name[i][j]] = {
                    'base acc': base_acc[i][j],
                    'adv acc': adv_acc[i][j]
                }
    for model_name in info.keys():
        print(model_name)
        print(info[model_name])
