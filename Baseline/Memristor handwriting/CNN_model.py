import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # 定义卷积层部分
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  # 第一层卷积，输入1个通道，输出32个通道
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化层，2x2的窗口
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # 第二层卷积，输入32个通道，输出64个通道
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层，2x2的窗口
        )
        
        # 定义全连接层部分
        self.fc_net = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),  # 全连接层，输入是展平后的64个通道 * 7 * 7的特征图
            nn.ReLU(),  # 激活函数
            nn.Linear(128, 10)  # 最后一层全连接层，输出10个类别
        )

    def forward(self, x):
        # 输入经过卷积层部分
        x = self.conv_net(x)
        
        # 展平特征图
        x = x.view(x.size(0), -1)  # 将 (batch_size, 64, 7, 7) 展平成 (batch_size, 64*7*7)
        
        # 经过全连接层
        x = self.fc_net(x)
        return x
