import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet #从model.py导入


def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)), #transforms.Resize((32,32))因为下载的图不一定是标准的，先转为32,32
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet() #实例化
    net.load_state_dict(torch.load('Lenet.pth')) #载入权重文件

    im = Image.open('luxing_cat.jpg') #通过PIL、numpy一般导入的格式为（height,width,channel）[H,W,C]
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # 加上batch维度（dim=0表示在最前面） [N, C, H, W]

    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].data.numpy()
    print(classes[int(predict)])

    #使用softmax
    with torch.no_grad():
        outputs = net(im)
        predict = torch.softmax(outputs, dim = 1) #dim=0是batch维度
    print(predict)


if __name__ == '__main__':
    main()
