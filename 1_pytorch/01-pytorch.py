import torch
import torch.nn as nn
import torch.nn.functional as F



#定义网络类
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #定义第一层卷积层，输入维度为1，输出维度为6，卷积核大小3*3
        self.conv1 = nn.Conv2d(1,6,3)
        #定义第二层，输入维度=6，输出维度=16，卷积核大小3*3
        self.conv2 = nn.Conv2d(6,16,3)
        #定义三层全连接神经网络
        self.fc1 = nn.Linear(16*6*6,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        #注意：任意卷积层后面要加激活层、池化层
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        #经过卷积层的处理后，张量要进入全连接层，进入前需调整张量的形状
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 
    
    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
net = Net()
#print(net)

params = list(net.parameters())
#print(len(params))
#print(params[0].size())

input = torch.randn(1,1,32,32)
#out = net(input)
# print(out)
# print(out.size())

target = torch.randn(10)
target = target.view(1,-1)
criterion = nn.MSELoss()

#loss = criterion(out,target)
#print(loss)


#pytorch 中首先执行梯度清零的操作
# net.zero_grad()

# # print('conv1.bias.grad before backward.')
# # print(net.conv1.bias.grad)

# #在pytoch中实现一次反向传播
# loss.backward()

# print('conv1.bias.grad after backward.')
# print(net.conv1.bias.grad)


#第一步导入优化器包
import torch.optim as optim

#构建优化器
optimizer = optim.SGD(net.parameters(),lr=0.01)

#第二步将优化器清零
optimizer.zero_grad()

#第三步执行网络计算并计算损失值
output = net(input)
loss = criterion(output,target)

#第四步执行反向传播
loss.backward()

#第五步更新参数
optimizer.step()
