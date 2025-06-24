import torch.nn as nn

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                     # 展平输入
            nn.Linear(784, 128, bias=False),  # 第一层，全连接层，输入为784，输出为128
            nn.ReLU(),                        # 激活函数ReLU，用于引入非线性
            nn.Dropout(0.5), 
            nn.Linear(128, 64, bias=False),   # 第二层，全连接层，输入为128，输出为64
            nn.ReLU(),                        # 激活函数ReLU
            nn.Dropout(0.5), 
            nn.Linear(64, 10, bias=False)     # 第三层，全连接层，输入为64，输出为10（对应10个类别）
        )

    def forward(self, x):
        return self.net(x)
