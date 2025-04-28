import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 10,bias=False)  # 定义全连接层
        )

    def forward(self, x):
        return self.net(x)