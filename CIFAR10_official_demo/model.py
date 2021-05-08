import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__() #继承nn.Module
        self.conv1 = nn.Conv2d(3, 16, 5) #input(3*32*32) 经卷积后的矩阵为（32-5+2*0）/1+1 = 28 所以下面out（16,28,28）16是卷积核的个数
        #3是in_channel，输入的彩色图片代表RGB三个分量，out_channels：对应卷积核个数,kernel_size=5代表卷积核5*5，默认stride=1
        #定义下采样层maxpool
        self.pool1 = nn.MaxPool2d(2, 2)#第一个参数kernel_size, 第二个参数是stride（默认=kernel_size）
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120) #后面的120是根据LeNet(1998)提出的，可以改
        self.fc2 = nn.Linear(120, 84) #同上
        self.fc3 = nn.Linear(84, 10) #最后的10是10个类别


    def forward(self, x): #x就是输入的数据[batch, channle, height, width]
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14) 池化层不改变深度，只改变高宽
        x = F.relu(self.conv2(x))    # output(32, 10, 10) 32是conv2中卷积核的个数，10 = （14-5+0）/1+1
        x = self.pool2(x)            # output(32, 5, 5)，然后送入全连接层，因为全连接层的输入为1维向量
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x

        #在训练网络过程中，计算卷积交叉熵过程中，内置了softmax所以这里不用加
#测试
#import torch
#input1 = torch.rand([32,3,32,32])
#model = LeNet()
#print(model)
#output = model(input1)

