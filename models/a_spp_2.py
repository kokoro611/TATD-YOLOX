import torch
import torch.nn as nn
from .network_blocks import BaseConv, get_activation
import torch.nn.functional as F
from torchviz import make_dot
from tensorboardX import SummaryWriter


class Get_rate_se(nn.Module):
    def __init__(self, in_channels, out_rate_num=3, activation='silu'):
        super(Get_rate_se, self).__init__()
        self.Gloab_Pool = nn.AdaptiveAvgPool2d((1,1))

        self.Linear_1 = nn.Linear(in_channels, int(in_channels // 2))
        self.Silu = get_activation(activation)
        self.Linear_2 = nn.Linear(int(in_channels // 2), out_rate_num)
        self.Softmax = nn.Softmax()

    def forward(self, x):
        x = self.Gloab_Pool(x)
        x = x.view(x.shape[0], -1)
        x = self.Linear_1(x)
        x = self.Softmax(x)
        x = self.Linear_2(x)
        x = self.Softmax(x)

        x = x[0]
        x = x * 20
        for i in range(len(x)):
            x[i] = int(x[i])
            if x[i] % 2 == 0:
                x[i] = x[i] - 1

        return x



'''
class Get_rate(nn.Module):
    def __init__(self, in_channles, in_size, out_rate_num=3, activation='silu'):
        super(Get_rate, self).__init__()
        self.in_size = in_size

        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.Conv1 = BaseConv(in_channels=in_channles, out_channels=in_channles, ksize=3, stride=2, act=activation)

        self.Linear_1 = nn.Linear(((in_size // 4) ** 2 * in_channles), (((in_size // 4) ** 2 * in_channles) // 2))
        self.Linear_2 = nn.Linear((((in_size // 4) ** 2 * in_channles) // 2), out_rate_num)

        self.Softmax_1 = nn.Softmax()
        self.Softmax_2 = nn.Softmax()

    def forward(self, x):
        x = self.MaxPool(x)
        x = self.Conv1(x)

        x = x.view(x.shape[0], -1)
        x = self.Linear_1(x)
        x = self.Softmax_1(x)
        x = self.Linear_2(x)
        x = self.Softmax_2(x)

        x = x[0]
        x = x * (self.in_size // 2)
        for i in range(len(x)):
            x[i] = int(x[i])
            if x[i] % 2 == 0:
                x[i] = x[i] - 1
        return x
'''

class SPP(nn.Module):
    def __init__(self, in_channles, out_channles, activation='silu'):
        super(SPP, self).__init__()
        hidden_channles = in_channles // 2
        self.Conv1 = BaseConv(in_channels=in_channles, out_channels=hidden_channles, ksize=1, stride=1, act=activation)

        Conv2_channles = hidden_channles * 4
        self.Conv2 = BaseConv(in_channels=Conv2_channles, out_channels=out_channles, ksize=1, stride=1, act=activation)

    def forward(self, x, r1, r2, r3):
        x = self.Conv1(x)
        x_res = x

        pad_1 = (r1 - 1) // 2
        pad_2 = (r2 - 1) // 2
        pad_3 = (r3 - 1) // 2

        x_pool_1 = F.max_pool2d(input=x, kernel_size=(r1, r1), stride=1, padding=pad_1)
        x_pool_2 = F.max_pool2d(input=x, kernel_size=(r2, r2), stride=1, padding=pad_2)
        x_pool_3 = F.max_pool2d(input=x, kernel_size=(r3, r3), stride=1, padding=pad_3)

        x_1 = torch.cat([x_res, x_pool_1], dim=1)
        x_2 = torch.cat([x_pool_2, x_pool_3], dim=1)
        x_3 = torch.cat([x_1, x_2], dim=1)

        x_3 = self.Conv2(x_3)

        return x_3


class Adaptive_SPP(nn.Module):
    def __init__(self, in_channles, out_channles):
        super(Adaptive_SPP, self).__init__()
        self.in_channles = in_channles
        self.out_channles = out_channles
        self.get_rate = Get_rate_se(in_channles, out_rate_num=3, activation='silu')
        self.spp = SPP(self.in_channles, self.out_channles)

    def forward(self, x):
        rate = self.get_rate(x)
        x = self.spp(x, int(rate[0]), int(rate[1]), int(rate[2]))
        return x

from thop import profile
if __name__ == "__main__":
    x = torch.rand((1, 256, 20, 20))
    net = Adaptive_SPP(256, 256)
    y = net(x)
    print(net)
    # g = make_dot(y)
    # g.render('espnet_model', view=False)
    #
    # with SummaryWriter(comment='Get_rate') as w:
    #     w.add_graph(net, x)
    flops,params = profile(net, inputs=(x,))
    print('FLOPs='+str(flops*2/1000**3)+'G')