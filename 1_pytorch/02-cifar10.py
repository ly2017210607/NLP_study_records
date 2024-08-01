import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms

#todo1:数据集下载
#数据转换器
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

#/data/ly23/pytorch_classfiy
trainset = torchvision.datasets.CIFAR10(root='/data/ly23/pytorch_classfiy',train=True,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/data/ly23/pytorch_classfiy',train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

import numpy as np
import matplotlib.pyplot as plt

#构建展示图片的函数
def imshow(img):
    img = img/2 +0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

#从数据迭代器中读取一张图片
# dataiter = iter(trainloader)
# images, labels = next(dataiter)

# #展示图片
# imshow(torchvision.utils.make_grid(images))
# #打印标签
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

#todo2:构建神经网络分类
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #定义两个卷积层，输入通道是3，输出通道是6，卷积核是5*5
        self.conv1 = nn.Conv2d(3,6,5)
        #定义卷积层，输入通道是6，输出通道是16，卷积核是5*5
        self.conv2 = nn.Conv2d(6,16,5)
        #定义池化层
        self.pool = nn.MaxPool2d(2,2)
        #定义三个全连接层
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #变换x的形状以适配全连接层的输入
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x 


net = Net()
#print(net)

#todo3:定义损失函数和优化器
import torch.optim as optim
#定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()
#定义优化器，随机梯度下降优化器
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

#todo4:训练神经网络
for epoch in range(2):
    running_loss = 0.0
    #按批次迭代训练模型
    for i, data in enumerate(trainloader,0):
        #从data中取出含有输入图像的inputs、标签张量labels
        inputs,labels = data

        #step1:梯度清零
        optimizer.zero_grad()
        #step2:将输入图像进入网络中并得到输出张量
        outputs = net(inputs)
        #计算损失值
        loss = criterion(outputs,labels)
        #step3:反向传播和梯度更新
        loss.backward()
        optimizer.step()

        #终端上打印训练的信息:轮次和损失值
        running_loss += loss.item()
        if (i + 1) % 2000 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

#设定模型的保存位置
PATH = './cifar_net.pth'
#保存模型的状态字典
torch.save(net.state_dict(),PATH)

#todo5:在测试集上测试模型
dataiter = iter(testloader)
images,labels = next(dataiter)

#打印原始图片
imshow(torchvision.utils.make_grid(images))
#打印真实的标签
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

#实例化模型的类对象
net = Net()

#加载训练阶段保存好的状态字典
net.load_state_dict(torch.load(PATH))

#利用模型对图片进行预测
outputs = net(images)
#模型有10个类别的输出，选取其中概率最大的类别作为预测值
_, predicted = torch.max(outputs,1)

#打印预测标签
print('Predicted: ', ''.join('%5s' %classes[predicted[j]] for j in range(4)))


#在整个测试集中测试模型的准确率
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('10000测试集的准确率: %d %%' %(100*correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in testloader:
        images, labels = data
        # ========
        # net.to(device)
        # inputs, labels = data[0].to(device), data[1].to(device)
        # ========
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        # 先看torch.squeeze() 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度，比如是一行或者一列这种，
        # 一个一行三列（1,3）的数去掉第一个维数为一的维度之后就变成（3）行
        c = (predicted == labels)
        print('c=====', c.size())
        c = (predicted == labels).squeeze()
        for i in range(4):
            # print('labels===', labels)
            # print('c===', c)
            label =  labels[i]
            # print("label===", label)
            # print('c[]===', c[i])
            # print('c[]===', c[i].item())
            '''
            labels=== tensor([9, 3, 4, 4])
            c=== tensor([ True,  True, False, False])
            label=== tensor(9)
            c[]=== tensor(True)
            c[]=== True
            labels=== tensor([9, 3, 4, 4])
            c=== tensor([ True,  True, False, False])
            label=== tensor(3)
            c[]=== tensor(True)
            c[]item=== True
            '''
            class_correct[label] += c[i].item()  # 这里只把是True的类别进行统计
            class_total[label] += 1

        # print("class_correct===", class_correct)
        # print("class_total===", class_total)

for i in range(10):
    # print("class_correct[i]===", class_correct[i])
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))