import torch
import torch.nn as nn
from torch.autograd import Variable

# 定义一个简单的三层全连接神经网络HandModel
class HandModel(nn.Module):
    def __init__(self):
        super(HandModel, self).__init__()
        # 第一层全连接层，输入维度为48，输出维度为40
        self.linear1 = nn.Linear(48, 40)
        # 第二层全连接层，输入维度为40，输出维度为32
        self.linear2 = nn.Linear(40, 32)
        # 第三层全连接层，输入维度为32，输出维度为27
        self.linear3 = nn.Linear(32, 27)

    def forward(self, input):
        # 将输入数据转换为float32类型
        input = input.to(torch.float32)
        # 通过第一层全连接层
        out = self.linear1(input)
        # 通过第二层全连接层
        out = self.linear2(out)
        # 通过第三层全连接层
        out = self.linear3(out)
        # 返回最终输出
        return out

# 定义一个包含LSTM的神经网络DynamicHandModel，用于处理序列数据
class DynamicHandModel(nn.Module):
    def __init__(self):
        super(DynamicHandModel, self).__init__()
        # LSTM层，输入维度为30，隐藏层维度为256，包含2层
        self.lstm = nn.LSTM(30, 256, num_layers=2)
        # 全连接层，连接LSTM输出，输入维度为256，输出维度为3
        self.linear1 = nn.Linear(256, 3)

    def forward(self, input, hidden=None):
        # 将输入数据转换为float32类型
        input = input.to(torch.float32)
        # 如果没有提供隐藏状态，则初始化隐藏状态和细胞状态为零
        if hidden is None:
            h_0 = input.data.new(2, 1, 256).fill_(0).float()
            c_0 = input.data.new(2, 1, 256).fill_(0).float()
            h_0, c_0 = Variable(h_0), Variable(c_0)
        else:
            h_0, c_0 = hidden

        # 通过LSTM层，同时更新隐藏状态和细胞状态
        out, hidden = self.lstm(input, (h_0, c_0))

        # 通过全连接层，将LSTM的输出转换为最终的输出维度
        out = self.linear1(out.view(1, -1))

        # 返回最终输出和更新后的隐藏状态
        return out, hidden

# 主函数，用于打印网络结构
if __name__ == "__main__":
    print(HandModel())
    print(DynamicHandModel())
