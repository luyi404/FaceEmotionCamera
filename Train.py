import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
class DataSet(torch.utils.data.Dataset):
    def __init__(self, dataPath, transform=None):
        self.dataset = pd.read_csv(dataPath)
        self.dataset = self.dataset[["emotion", "pixels"]]  # 获取了相应的数据集了
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):  # 可以根据下表取数据
        emotion = self.dataset.loc[index, "emotion"]
        pixels = self.dataset.loc[index, "pixels"]  # 2304 = 48 * 48
        pixels = pixels.split(" ")
        pixels = list(map(float, pixels))
        pixels = torch.Tensor(pixels).reshape(48, 48)
        if self.transform:
            pixels = self.transform(pixels)
        # emotion_onehot = np.zeros(7,1)
        # emotion_onehot[emotion][0] = 1
        # emotion_onehot = torch.Tensor(emotion_onehot)
        return pixels, emotion


TheTransform = torchvision.transforms.Compose([  # 处理工作
    torchvision.transforms.ToPILImage(),  # 转化为PIL Image
    # torchvision.transforms.Grayscale(),
    # torchvision.transforms.Resize(224),
    torchvision.transforms.RandomHorizontalFlip(),  # 随机翻转一下
    torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5),  # 随机调整亮度和对比度
    torchvision.transforms.ToTensor(),  # 再变回tensor
])


def vgg_block(num_convs, in_channels, out_channels):#定义产生VGG块的东西
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    # 宽高减半
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)

class FlattenLayer(nn.Module):  #设置个摊平的
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x #按照我的dataset的话应该是(batch_size,emotion,pixels)的样子？

def VGG(conv_arch, fc_features, fc_hidden_units = 4096):
    net = nn.Sequential()
    for i,(num_convs, in_channels, out_channels) in enumerate(conv_arch):   #加VGG块了
        net.add_module("vggBlock_" + str(i + 1), vgg_block(num_convs, in_channels, out_channels))
    net.add_module("fc", nn.Sequential(FlattenLayer(),
                                       nn.Linear(fc_features, fc_hidden_units),
                                       nn.ReLU(), #使用LeajyReLu,虽然不确定到底好不好用，但是感觉比ReLu高级一些
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, 7))) #输出成7个感情
    return net

Trainloss = []
TrainAcc = []
ValAcc = []
def train(train_iter, test_iter, net, optimzer, device, num_epochs):
    axis_x = range(1, num_epochs + 1) #画图用
    net = net.to(device)
    loss = nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(1, num_epochs + 1):
        train_loss_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time() #计时
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            # print(X.shape)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimzer.zero_grad()
            l.backward()
            optimzer.step()
            train_loss_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        ValAcc.append(test_acc)
        TrainAcc.append(train_acc_sum / n)
        Trainloss.append(train_loss_sum / batch_count)
        print('epoch {}, loss {:.4f}, train acc {:.4f}, test acc{:.4f}, time {:.2f} sec'
          .format(epoch, train_loss_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        torch.save(model, '/content/drive/My Drive/Emotion/model' + str(epoch) +".pth")
    plt.plot(axis_x,Trainloss,label = "TrainLoss",color = "r" )
    plt.plot(axis_x,TrainAcc,label = "TrainAcc",color = "b" )
    plt.plot(axis_x,ValAcc,label = "ValAcc",color = "g" )
    plt.xlabel('Epoch')
    plt.title('Result of my train')
    plt.legend()
    plt.show()

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                # 评估模式, 关闭dropout
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                # 改回训练模式
                net.train()
            else:
                if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

if __name__ == '__main__':  #不加这个不可以多进程读取DataSet
    DEVICE = "cuda"
    batch_size = 64
    dataset_train = DataSet("/content/drive/My Drive/Emotion/data2/data/Train.csv", transform=TheTransform)
    dataset_vali = DataSet("/content/drive/My Drive/Emotion/data2/data/Val.csv", transform=TheTransform)
    DataLoader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=2)
    DataLoader_vali = torch.utils.data.DataLoader(dataset=dataset_vali,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2)
    conv_arch = ((2, 1, 32), (2, 32, 64), (1, 64, 128))
    # 经过3个vgg_block, 宽高会减半3次, 变成 48 / 8 = 6
    fc_features = 128 * 6 * 6  # c * w * h 128是进过VGG后的通道数
    fc_hidden_units = 1024
    num_epochs = 30
    model = VGG(conv_arch, fc_features, fc_hidden_units).to(DEVICE)
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train(DataLoader_train, DataLoader_vali, model, optimizer, DEVICE, num_epochs)