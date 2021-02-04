import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import *
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import torch
import numpy as np
from torch.autograd import Variable
from imagecorruptions import corrupt
import skimage as sk
from skimage.filters import gaussian

device = torch.device('cuda:4')


def gaussian_noise(x, severity=1):
    c = [0.04, 0.06, .08, .09, .10][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.01, .02, .03, .05, .07][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.05, 1, 1), (0.25, 1, 1), (0.4, 1, 1), (0.25, 1, 2), (0.4, 1, 2)][severity - 1]

    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)
    # print(x.shape)
    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(32 - c[1], c[1], -1):
            for w in range(32 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[:, :, h, w], x[:, :, h_prime, w_prime] = x[:, :, h_prime,  w_prime], x[:, :, h, w]

    return np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255


def contrast(x, severity=1):
    c = [.75, .5, .4, .3, 0.15][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(2, 3), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def LoadMNIST(root, transform, batch_size, download=True):
    train_dataset = datasets.MNIST(root=root, train=True, transform=transform, download=download)
    test_dataset = datasets.MNIST(root=root, train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 150)
        self.fc3 = nn.Linear(150, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    import Resnet18

    net = Resnet18.getResNet18()
    SoftmaxWithXent = nn.CrossEntropyLoss()
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
    train_dataset = datasets.CIFAR10(root='../../LR/EBP', train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root='../../LR/EBP', train=False, transform=transform_test, download=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # for train
    trainLoss = 0.
    testLoss = 0.
    learning_rate = 1e-3
    start_epoch = 0
    test_loss_list = []
    train_loss_list = []
    acc_list = []
    epoches = 30
    net = net.to(device)
    # define optimization algorithm
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-04)
    if not os.path.exists('./mnist_model/black_Resnet18.pth'):
        for epoch in range(start_epoch, start_epoch + epoches):
            train_N = 0.
            train_n = 0.
            trainLoss = 0.
            net.train()
            for batch, [trainX, trainY] in enumerate(tqdm(train_dataloader, ncols=10)):
                train_n = len(trainX)
                train_N += train_n
                trainX = trainX.to(device)
                trainY = trainY.to(device).long()
                optimizer.zero_grad()
                predY = net(trainX)
                loss = SoftmaxWithXent(predY, trainY)

                loss.backward()  # get gradients on params
                optimizer.step()  # SGD update
                trainLoss += loss.detach().cpu().numpy()
            trainLoss /= train_N
            train_loss_list.append(trainLoss)
            test_N = 0.
            testLoss = 0.
            correct = 0.
            net.eval()
            for batch, [testX, testY] in enumerate(tqdm(test_dataloader, ncols=10)):
                test_n = len(testX)
                test_N += test_n
                testX = testX.to(device)
                testY = testY.to(device).long()
                predY = net(testX)
                loss = SoftmaxWithXent(predY, testY)
                testLoss += loss.detach().cpu().numpy()
                _, predicted = torch.max(predY.data, 1)
                correct += (predicted == testY).sum()
            testLoss /= test_N
            test_loss_list.append(testLoss)
            acc = correct / test_N
            acc_list.append(acc)
            print('epoch:{} train loss:{} testloss:{} acc:{}'.format(epoch, trainLoss, testLoss, acc))
        if not os.path.exists('./mnist_model'):
            os.mkdir('mnist_model')
        if not os.path.exists('./mnist_logs'):
            os.mkdir('mnist_logs')
        torch.save(net.state_dict(), './mnist_model/black.pth')
        print('模型已经保存')
    else:
        static = torch.load('./mnist_model/black_Resnet18.pth', map_location='cpu')
        net.load_state_dict(static)
    import pickle as pkl

    with open('./mnist_logs/black.pkl', 'wb') as file:
        pkl.dump([train_loss_list, test_loss_list, acc_list], file)

    net = net.to(torch.device('cpu'))
    xs = []
    y_trues = []
    for data in tqdm(test_dataloader):
        inputs, labels = data
        if len(xs) == 0:
            xs = inputs
            y_trues = labels
        else:
            xs = torch.cat([xs, inputs], dim=0)
            y_trues = torch.cat([y_trues, labels], dim=0)
    xs = np.array(xs)
    y_trues = np.array(y_trues).reshape(-1)

    # noise = ['gaussian', 'impulse', 'glass_blur', 'contrast', 'FGSM', 'LBFGS']
    # noise_function = [gaussian_noise, impulse_noise, glass_blur, contrast]
    noise = ['FGSM', 'LBFGS']
    noise_function = [glass_blur, contrast]
    strength = [1, 2, 3, 4, 5]
    epsilon = 0.1
    for i in range(len(noise)):
        noises = []
        y_preds = []
        y_preds_adversarial = []
        totalMisclassifications = 0
        xs_clean = []
        y_trues_clean = []
        num_adv = 0
        N = 0

        if noise[i] == 'FGSM':
            net = net.to(device)
            for x, y_true in tqdm(test_dataloader):
                x = x
                # Wrap x as a variable
                x = Variable(x.to(device), requires_grad=True)
                x = x.to(device)
                y_true = Variable(torch.LongTensor(np.array(y_true)), requires_grad=False)
                y_true = y_true.to(device)
                # Classification before Adv
                y_pred = np.argmax(net(x).cpu().data.numpy(), axis=-1)
                # Generate Adversarial Image
                # Forward pass
                outputs = net(x)
                loss = SoftmaxWithXent(outputs, y_true)
                loss.backward()  # obtain gradients on x

                # Add perturbation
                x_grad = torch.sign(x.grad.data)
                x_adversarial = torch.clamp(x.data + epsilon * x_grad, 0, 1)

                # Classification after optimization
                y_pred_adversarial = np.argmax(net(Variable(x_adversarial)).cpu().data.numpy(), axis=-1)
                # print "Before: {} | after: {}".format(y_pred, y_pred_adversarial)

                totalMisclassifications += np.array(y_true.cpu().data.numpy() != y_pred).sum()
                num_adv += np.array(y_pred_adversarial != y_pred).sum()

                y_preds.append(y_pred)
                y_preds_adversarial.append(y_pred_adversarial)
                noises.append(x_adversarial.cpu().numpy())
                xs_clean.append(x.cpu().data.numpy())
                y_trues_clean.append(y_true.cpu().data.numpy())
                N += 1

            print("Total totalMisclassifications :{}/{} ".format(totalMisclassifications, len(xs)))  # 1221/1797
            print("the amount of adv samples is : {}".format(num_adv))  # 576

            print("Successful!!")

            with open("Generate_adversarial_sample_epsilon=" + str(epsilon) + "by_FGSM.pkl", "wb") as f:
                adv_data_dict2 = {
                    "xs": xs_clean,
                    "y_trues": y_trues_clean,
                    "y_preds": y_preds,
                    "noises": noises,
                    "y_preds_adversarial": y_preds_adversarial
                }
                pickle.dump(adv_data_dict2, f, protocol=3)
            print("Successful!!")
            continue
        elif noise[i] == 'LBFGS':
            net = net.to(device)
            for x, y_true in tqdm(test_dataloader):
                xs_clean.append(np.array(x))
                y_true = y_true
                # Wrap x as a variable
                xv = torch.Tensor(x).to(device)
                xv = nn.Parameter(xv, requires_grad=True)
                # xv = nn.Parameter(torch.FloatTensor(x.reshape(1, 28 * 28)), requires_grad=True)
                y_true = Variable(torch.LongTensor(np.array(y_true)).to(device), requires_grad=False)
                method = optim.LBFGS([xv], lr=5e-2)
                # Classification before Adv
                y_pred = np.argmax(net(xv).cpu().data.numpy(), axis=-1)


                # Generate Adversarial Image
                def closure():
                    method.zero_grad()
                    output = net(xv)
                    loss = -SoftmaxWithXent(output, y_true)
                    loss.backward()
                    return loss


                method.step(closure)
                # method = optim.LBFGS(list(xv), lr=1e-1)
                # Add perturbation
                # x_grad = torch.sign(x.grad.data)
                x_adversarial = torch.clamp(xv.data, 0, 1)

                # Classification after optimization
                y_pred_adversarial = np.argmax(net(Variable(x_adversarial)).cpu().data.numpy(), axis=-1)
                # print "Before: {} | after: {}".format(y_pred, y_pred_adversarial)

                # print "Y_TRUE: {} | Y_PRED: {}".format(_y_true, y_pred)
                totalMisclassifications += np.array(y_true.cpu().data.numpy() != y_pred).sum()
                num_adv += np.array(y_pred_adversarial != y_pred).sum()

                y_preds.append(y_pred)
                y_preds_adversarial.append(y_pred_adversarial)
                noises.append(x_adversarial.cpu().numpy())
                y_trues_clean.append(y_true.cpu().data.numpy())
                N += 1

            print("Total totalMisclassifications :{}/{} ".format(totalMisclassifications, len(xs)))  # 1221/1797
            print("the amount of adv samples is : {}".format(num_adv))  # 576

            print("Successful!!")

            with open("Generate_adversarial_sample_epsilon=" + str(epsilon) + "by_LBFGS.pkl", "wb") as f:
                adv_data_dict2 = {
                    "xs": xs_clean,
                    "y_trues": y_trues_clean,
                    "y_preds": y_preds,
                    "noises": noises,
                    "y_preds_adversarial": y_preds_adversarial
                }
                pickle.dump(adv_data_dict2, f, protocol=3)
            print("Successful!!")
            continue
        for j in range(len(strength)):
            noises = []
            y_preds = []
            y_preds_adversarial = []
            totalMisclassifications = 0
            xs_clean = []
            y_trues_clean = []
            num_adv = 0
            N = 0
            for x, y_true in tqdm(test_dataloader):
                # Wrap x as a variable
                x = x.reshape(-1, 3, 32, 32)
                xs_clean.append(np.array(x))
                xv = torch.Tensor(x)
                # xv = nn.Parameter(torch.FloatTensor(x.reshape(1, 28 * 28)), requires_grad=True)
                y_true = Variable(torch.LongTensor(np.array(y_true)), requires_grad=False)
                # Classification before Adv
                y_pred = 0

                # Generate Adversarial Image
                xv = xv.numpy() * 255
                # chw->hwc
                xv = noise_function[i](xv, strength[j]).astype(np.float)
                xv /= 255
                # hwc->chw
                xv = torch.from_numpy(xv).float()
                # method = optim.LBFGS(list(xv), lr=1e-1)
                # Add perturbation
                # x_grad = torch.sign(x.grad.data)
                x_adversarial = torch.clamp(xv, 0, 1)

                # Classification after optimization
                y_pred_adversarial = 0 # np.argmax(net(Variable(x_adversarial)).data.numpy())
                # print "Before: {} | after: {}".format(y_pred, y_pred_adversarial)

                # print "Y_TRUE: {} | Y_PRED: {}".format(_y_true, y_pred)
                # totalMisclassifications += np.array(y_true.data.numpy() != y_pred).sum()
                # num_adv += np.array(y_pred_adversarial != y_pred).sum()

                y_preds.append(y_pred)
                y_preds_adversarial.append(y_pred_adversarial)
                noises.append(x_adversarial.numpy())
                y_trues_clean.append(y_true.data.numpy())
                N += 1

            print("Total totalMisclassifications :{}/{} ".format(totalMisclassifications, len(xs)))  # 1221/1797
            print("the amount of adv samples is : {}".format(num_adv))  # 576

            print("Successful!!")
            with open("Generate_adversarial_sample_epsilon=" + str(epsilon) + "by_{}_{}.pkl".format(noise[i],
                                                                                                    strength[j]),
                      "wb") as f:
                adv_data_dict2 = {
                    "xs": xs_clean,
                    "y_trues": y_trues_clean,
                    "y_preds": y_preds,
                    "noises": noises,
                    "y_preds_adversarial": y_preds_adversarial
                }
                pickle.dump(adv_data_dict2, f, protocol=3)
            print("{}-{} Successful!!".format(noise[i], strength[j]))
