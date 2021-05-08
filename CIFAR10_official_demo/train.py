import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),  # ToTensor表示把image/numpy.ndarray to tensor
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 标准化,将值减去均值再除以标注差
    # transform是对图像进行预处理的函数

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=False, transform=transform)  # 下载到当前目录的data文件夹
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)  # 每一批随机拿出36张图，num_workers在Windows默认0

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                             shuffle=False, num_workers=0)

    val_data_iter = iter(val_loader)  # 转化为可迭代的迭代器
    val_image, val_label = val_data_iter.next()  # 获取一批数据

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




    net = LeNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 在GPU跑
    net.to(device)

    loss_function = nn.CrossEntropyLoss() #内置了softmax 不需要在网络输出加了
    optimizer = optim.Adam(net.parameters(), lr=0.001) #将模型可训练的参数都训练

    # if torch.cuda.is_available():
    #     print("CUDA is enable!")
    #     net = net.cuda()
    #     net.train()


    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        time_start = time.perf_counter()
        for step, data in enumerate(train_loader, start=0): #遍历训练集样本，返回data和step（步数 从0开始）
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data #将data分为输入图像和标签
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad() #如果不清除历史梯度，就会对计算的历史梯度进行累加（理论上batchsize越大越好但是硬件受限），通过这个特性变相实现很大batch训练
            # forward + backward + optimize
            outputs = net(inputs) #input 传入GPU
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:    # print every 500 mini-batches
                with torch.no_grad(): #with是上下文管理器，不计算每个节点的误差损失梯度（省算力资源）测试/预测要加这个函数
                    outputs = net(val_image.to(device))  # [batch, 10]
                    predict_y = torch.max(outputs, dim=1)[1] #寻找输出最大的index在什么位置
                    #上面dim=1因为 第0维度是batch,第1维度是从10个类别中找最大 [1]是找位置，predict_y即标签类别
                    accuracy = torch.eq(predict_y, val_label.to(device)).sum().item() / val_label.size(0)
                    #accuracy = (predict_y == val_label).sum().item() / val_label.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy)) #epoch：训练到第几轮
                    print('%f s' % (time.perf_counter() - time_start))
                    running_loss = 0.0

    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path) #保存模型

if __name__ == '__main__':
    main()

