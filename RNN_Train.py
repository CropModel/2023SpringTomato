import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# 假设您已经读取并处理了Excel数据，其中每行对应一天的数据
# 将数据按照植物编号进行分组

def load_data(filename):
    # 转换为 PyTorch 张量
    data = np.load(filename)
    data = torch.from_numpy(data).float()
    return data

# 定义 RNN 模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])

        return out

Predata_train = load_data("TrainCode/Data/PreData30-12_train.npy")
Predata_val   = load_data("TrainCode/Data/PreData30-12_val.npy")
labels_train  = load_data("TrainCode/Data/LabelData30-12_train.npy")
labels_val    = load_data("TrainCode/Data/LabelData30-12_val.npy")


for  n in [15,20,25,30]:

    # 设置训练超参数
    input_size = 34  # 输入特征的维度
    hidden_size = n # RNN 隐含层的维度
    num_layers = 3  # RNN 的层数
    output_size = 12  # 输出标签的维度
    batch_size = 16  # 批次大小
    num_epochs = 10000  # 训练轮数
    learning_rate = 0.001  # 学习率

    # # 创建 RNN 模型实例
    model = RNN(input_size, hidden_size, num_layers, output_size)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 创建记录loss的文件
    file_path = "loss_l={0}_h={1}_out={2}-10000.txt".format(num_layers,hidden_size,output_size)  # 替换成您要操作的文件的路径
    # 训练模型
    for epoch in range(num_epochs):
        for i in range(0, len(Predata_train), batch_size):
            # 获取当前批次的数据和标签
            inputs = Predata_train[i:i+batch_size]
            targets = labels_train[i:i+batch_size]

            # 向前传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 打印每轮训练的损失
        with open(file_path, 'a') as file:
                file.write(f'{epoch+1}, {loss.item():.4f}\n')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), '30-12_{0}_{1}.pth'.format(num_layers,hidden_size))

    # 在测试集上进行预测
    # 载入已训练好的模型
    # model.load_state_dict(torch.load('model_vector14_1.pth'))
    model.eval()
    with torch.no_grad():
        test_data = load_data("TrainCode/Data/PreData30-12_val.npy")
        test_labels = load_data("TrainCode/Data/LabelData30-12_val.npy")
        test_outputs = model(test_data)
        loss = criterion(test_outputs, test_labels)
        # 计算每个样本的百分比误差
        percentage_errors = torch.abs((test_outputs - test_labels) / test_labels) * 100.0
        # 计算平均百分比误差
        mean_percentage_error = torch.mean(percentage_errors)
        print(loss.item(),mean_percentage_error)
        log_path = "logl{0}30-12-10000.txt".format(num_layers)
        with open(log_path, 'a') as file:
                file.write(f'{num_layers},{hidden_size},{loss.item():.4f},{mean_percentage_error}\n')