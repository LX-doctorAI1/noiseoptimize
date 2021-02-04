import models
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

device = models.device

if __name__ == '__main__':

    """dense_model = [models.DenseReLU, models.DenseSigmoid, models.DensePlusReLU, models.DensePlusSigmoid,
                   models.DenseNoiseReLU, models.DenseNoiseSigmoid]
    cnn_model = [models.CNN, models.CNNFCplus, models.CNNAplus, models.CNNFCN, models.CNNAN]

    dense_model_name = ['models.DenseReLU', 'models.DenseSigmoid', 'models.DensePlusReLU', 'models.DensePlusSigmoid',
                        'models.DenseNoiseReLU', 'models.DenseNoiseSigmoid']
    cnn_model_name = ['models.CNN', 'models.CNNFCplus', 'models.CNNAplus', 'models.CNNFCN', 'models.CNNAN']
    
    model_zoo_name = [dense_model_name, cnn_model_name]

    model_zoo = [dense_model]"""

    dense_model = [models.DenseReLU, models.DenseSigmoid, models.DensePlusReLU, models.DensePlusSigmoid,
                   models.DenseNoiseReLU, models.DenseNoiseSigmoid]
    cnn_model = [models.CNNAN]

    dense_model_name = ['models.DenseReLU', 'models.DenseSigmoid', 'models.DensePlusReLU', 'models.DensePlusSigmoid',
                        'models.DenseNoiseReLU', 'models.DenseNoiseSigmoid']
    cnn_model_name = ['models.CNNAN']

    model_zoo_name = [[], cnn_model_name]

    model_zoo = [[], cnn_model]

    transform = transforms.Compose([transforms.ToTensor()])
    batch_size = 128
    train_dataloader, test_dataloader = models.LoadMNIST('../../LR/data/MNIST', transform, batch_size, False)
    for i in range(len(model_zoo)):
        for j in range(len(model_zoo[i])):
            print('new model training:{}'.format(model_zoo_name[i][j]))
            model = model_zoo[i][j]()
            model = model.to(device)
            trainLoss = 0.
            testLoss = 0.
            learning_rate = 1e-3
            start_epoch = 0
            test_loss_list = []
            train_loss_list = []
            acc_list = []
            epoches = 10
            SoftmaxWithXent = nn.CrossEntropyLoss()
            # define optimization algorithm
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-04)
            print('{} epoch to run:{} learning rate:{}'.format(model_zoo_name[i][j], epoches, learning_rate))
            for epoch in range(start_epoch, start_epoch + epoches):
                train_N = 0.
                train_n = 0.
                trainLoss = 0.
                model.train()
                for batch, [trainX, trainY] in enumerate(tqdm(train_dataloader, ncols=10)):
                    train_n = len(trainX)
                    train_N += train_n
                    trainX = trainX.to(device)
                    if i == 0:
                        trainX = trainX.reshape(-1, 28 * 28)
                    trainY = trainY.to(device).long()
                    optimizer.zero_grad()
                    predY = model(trainX)
                    loss = SoftmaxWithXent(predY, trainY)

                    loss.backward()  # get gradients on params
                    optimizer.step()  # SGD update
                    trainLoss += loss.detach().cpu().numpy()
                trainLoss /= train_N
                train_loss_list.append(trainLoss)
                test_N = 0.
                testLoss = 0.
                correct = 0.
                model.eval()
                for batch, [testX, testY] in enumerate(tqdm(test_dataloader, ncols=10)):
                    test_n = len(testX)
                    test_N += test_n
                    testX = testX.to(device)
                    if i == 0:
                        testX = testX.reshape(-1, 28 * 28)
                    testY = testY.to(device).long()
                    predY = model(testX)
                    loss = SoftmaxWithXent(predY, testY)
                    testLoss += loss.detach().cpu().numpy()
                    _, predicted = torch.max(predY.data, 1)
                    correct += (predicted == testY).sum()
                testLoss /= test_N
                test_loss_list.append(testLoss)
                acc = correct / test_N
                acc_list.append(acc)
                print('epoch:{} train loss:{} testloss:{} acc:{}'.format(epoch, trainLoss, testLoss, acc))
                if acc > 0.97:
                    break
            if not os.path.exists('./mnist_model'):
                os.mkdir('mnist_model')
            if not os.path.exists('./mnist_logs'):
                os.mkdir('mnist_logs')
            torch.save(model.state_dict(), './mnist_model/{}.pth'.format(model_zoo_name[i][j]))
            print('模型已经保存')
            with open('./mnist_logs/{}.pkl'.format(model_zoo_name[i][j]), 'wb') as file:
                pkl.dump([train_loss_list, test_loss_list, acc_list], file)
