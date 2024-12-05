# import numpy as np
# import torch as t
# from model import HandModel
# from torch import nn
# from torchnet import meter
# from torch.autograd import Variable
# import copy
#
# # 定义一个标签列表，包含了一系列描述性的词汇
# # label = ["also", "attractive", "beautiful", "believe", "de", "doubt", "dream", "express", "eye", "give", "handLang",
# #          "have",
# #          "many",
# #          "me", "method", "no", "only", "over", "please", "put", "say", "smile", "star", "use_accept_give", "very",
# #          "watch",
# #          "you"]
# label = ["think", "why", "here", "home", "even", "look", "life", "and", "same",
#          "hope", "love",
#          "accept", "tell",
#          "give_up", "continue", "achieve",
#          "vigilant", "lie"]
# # label = ["why", "love", "think"]
# # 计算标签的数量
# label_num = len(label)
# print("label num:" + str(label_num))
# # 初始化目标标签数组，用于生成训练模型所需的标签数据
# targetX = [0 for xx in range(label_num)]
# target = []
#
# # 生成独热编码（One-Hot Encoding）形式的标签数据
# for xx in range(label_num):
#     # 深拷贝一个全零数组，以避免列表之间的引用影响
#     target_this = copy.deepcopy(targetX)
#     # 将当前标签位置设置为1，其余位置保持为0
#     target_this[xx] = 1
#     # 将处理后的标签数组添加到目标列表中
#     target.append(target_this)
#
# # 定义学习率和模型保存路径
# lr = 1e-3  # learning rate
# model_saved = 'checkpoints/model'
#
# # 模型定义
# model = HandModel()
# # 使用Adam优化器初始化模型参数，设置学习率
# optimizer = t.optim.Adam(model.parameters(), lr=lr)
# # 使用交叉熵损失函数
# criterion = nn.CrossEntropyLoss()
# # 初始化损失值的平均值计算器
# loss_meter = meter.AverageValueMeter()
# # 训练轮数
# epochs = 40
#
# # 迭代训练模型的主循环
# for epoch in range(epochs):
#     print("epoch:" + str(epoch))
#     loss_meter.reset()
#     count = 0
#     allnum = 0
#     # 遍历标签列表以加载对应数据
#     for i in range(len(label)):
#         data = np.load('./npz_files/' + label[i] + ".npz", allow_pickle=True)
#         data = data['data']
#         # print(data)
#         # 遍历当前数据集中的每个数据样本
#         for j in range(len(data)):
#             xdata = t.tensor(data[j])
#             optimizer.zero_grad()
#             this_target = t.tensor(target[i]).float()
#             print(this_target.shape)
#             input_, this_target = Variable(xdata), Variable(this_target)
#
#             output = model(input_)
#             # 打印模型输出的最大值索引
#             max_index = output.argmax().item()  # 使用argmax获取最大值索引
#             print("Max index:", max_index)
#             outLabel = label[output.tolist().index(max(output))]
#
#             targetIndex = target[i].index(1)
#             targetLabel = label[targetIndex]
#             if targetLabel == outLabel:
#                 count += 1
#             allnum += 1
#
#             # 为输出和目标张量添加维度以匹配形状
#             output = t.unsqueeze(output, 0)
#             this_target = t.unsqueeze(this_target, 0)
#
#             # 计算损失并进行反向传播
#             loss = criterion(output, this_target)
#             loss.backward()
#             optimizer.step()
#             loss_meter.add(loss.data)
#
#     # 打印当前epoch的正确率
#     print("correct_rate:", str(count / allnum))
#
#     # 保存当前epoch的模型参数
#     t.save(model.state_dict(), '%s_%s.pth' % (model_saved, epoch))
import numpy as np
import torch as t
from model import HandModel
from torch import nn
from torchnet import meter
from torch.autograd import Variable
import copy

# 定义一个标签列表，包含了一系列描述性的词汇
label = ["think", "why", "here", "home", "even", "look", "life", "and", "same",
         "hope", "love",
         "accept", "tell",
         "give_up", "continue", "achieve",
         "vigilant", "lie"]
label_num = len(label)
print("label num:" + str(label_num))

# 定义学习率和模型保存路径
lr = 1e-3  # learning rate
model_saved = 'checkpoints/model'

# 模型定义
model = HandModel()
optimizer = t.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
loss_meter = meter.AverageValueMeter()
epochs = 40

# 迭代训练模型的主循环
for epoch in range(epochs):
    print("epoch:" + str(epoch))
    loss_meter.reset()
    count = 0
    allnum = 0
    # 遍历标签列表以加载对应数据
    for i in range(len(label)):
        data = np.load('./npz_files/' + label[i] + ".npz", allow_pickle=True)
        data = data['data']
        # 遍历当前数据集中的每个数据样本
        for j in range(len(data)):
            xdata = t.tensor(data[j]).unsqueeze(0)  # 确保输入数据是2维的
            optimizer.zero_grad()
            this_target = t.tensor([i], dtype=t.long)  # 类别索引，1维张量
            input_, this_target = Variable(xdata), Variable(this_target)

            output = model(input_)
            max_index = output.argmax(dim=1).item()  # 使用argmax获取最大值索引
            print("Max index:", max_index)
            outLabel = label[max_index]

            targetIndex = i
            targetLabel = label[targetIndex]
            if targetLabel == outLabel:
                count += 1
            allnum += 1

            # 计算损失并进行反向传播
            loss = criterion(output, this_target)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.data)

    # 打印当前epoch的正确率
    print("correct_rate:", str(count / allnum))

    # 保存当前epoch的模型参数
    t.save(model.state_dict(), '%s_%s.pth' % (model_saved, epoch))