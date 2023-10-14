# import torch
# import torch.nn as nn
# import torch.optim as optim
import pandas as pd
import numpy as np
import os
# 假设您已经读取并处理了Excel数据，其中每行对应一天的数据
# 将数据按照植物编号进行分组

data = []  # 用于存储每株植物的序列数据
labels = []
train_data = []
# df = pd.read_excel('TrainCode/final_data2.xlsx').drop('高度', axis=1)
df = pd.read_excel('TrainCode/final_data2.xlsx')
offset = 0
data_size = 3276

# 将每7天的数据作为一组，其中前5天为输入序列，第6、7天为标签序列
# for i in range(0,data_size,7):
#         Input_data = df.iloc[i+offset:i+offset+5,:-2].values  # 输入序列为前5天的数据
#         Truehighth_of_input_data = [df.iloc[i+offset+0]["high"],df.iloc[i+offset+1]["high"],df.iloc[i+offset+2]["high"],df.iloc[i+offset+3]["high"],
#                     df.iloc[i+offset+4]["high"]]
#         Label_of_data= [df.iloc[i+offset+5]["high"],df.iloc[i+offset+6]["high"]]    # 标签序列为第6,7天的数据
#         data.append(Truehighth_of_input_data)
#         labels.append(Label_of_data)
#         train_data.append(Input_data)

# 将每14天的数据作为一组，其中前10天为输入序列，第11，12，13，14天为标签序列
# for i in range(0,data_size,14):
#         Input_data = df.iloc[i+offset:i+offset+10,:-2].values  # 输入序列为前10天的数据
#         Truehighth_of_input_data = [df.iloc[i+offset+0]["high"],df.iloc[i+offset+1]["high"],df.iloc[i+offset+2]["high"],df.iloc[i+offset+3]["high"],
#                     df.iloc[i+offset+4]["high"],df.iloc[i+offset+5]["high"],df.iloc[i+offset+6]["high"],df.iloc[i+offset+7]["high"],df.iloc[i+offset+8]["high"],
#                     df.iloc[i+offset+9]["high"]]
#         Label_of_data= [df.iloc[i+offset+10]["high"],df.iloc[i+offset+11]["high"],df.iloc[i+offset+12]["high"],df.iloc[i+offset+13]["high"]]    # 标签序列为第10,11,12,13天的数据
#         data.append(Truehighth_of_input_data)
#         labels.append(Label_of_data)
#         train_data.append(Input_data)

# 将每42天的数据作为一组，其中前30天为输入序列，第31-42天为标签序列
for i in range(0,data_size,42):
        Input_data = df.iloc[i+offset:i+offset+30,:-2].values  # 输入序列为前30天的数据
        Truehighth_of_input_data = [df.iloc[i+offset+0]["high"],df.iloc[i+offset+1]["high"],df.iloc[i+offset+2]["high"],df.iloc[i+offset+3]["high"],
                    df.iloc[i+offset+4]["high"],df.iloc[i+offset+5]["high"],df.iloc[i+offset+6]["high"],df.iloc[i+offset+7]["high"],df.iloc[i+offset+8]["high"],
                    df.iloc[i+offset+9]["high"],df.iloc[i+offset+10]["high"],df.iloc[i+offset+11]["high"],df.iloc[i+offset+12]["high"],df.iloc[i+offset+13]["high"],
                    df.iloc[i+offset+14]["high"],df.iloc[i+offset+15]["high"],df.iloc[i+offset+16]["high"],df.iloc[i+offset+17]["high"],df.iloc[i+offset+18]["high"],
                    df.iloc[i+offset+19]["high"],df.iloc[i+offset+20]["high"],df.iloc[i+offset+21]["high"],df.iloc[i+offset+22]["high"],df.iloc[i+offset+23]["high"],
                    df.iloc[i+offset+24]["high"],df.iloc[i+offset+25]["high"],df.iloc[i+offset+26]["high"],df.iloc[i+offset+27]["high"],df.iloc[i+offset+28]["high"],
                    df.iloc[i+offset+29]["high"]]
        Label_of_data= [df.iloc[i+offset+30]["high"],df.iloc[i+offset+31]["high"],df.iloc[i+offset+32]["high"],df.iloc[i+offset+33]["high"],
                    df.iloc[i+offset+34]["high"],df.iloc[i+offset+35]["high"],df.iloc[i+offset+36]["high"],df.iloc[i+offset+37]["high"],df.iloc[i+offset+38]["high"],
                    df.iloc[i+offset+39]["high"],df.iloc[i+offset+40]["high"],df.iloc[i+offset+41]["high"]]    # 标签序列为第31-42天的数据
        data.append(Truehighth_of_input_data)
        labels.append(Label_of_data)
        train_data.append(Input_data)

# 生成随机排列索引数组
random_indices = np.random.permutation(len(data))

# 使用随机排列索引数组来重排 data 和 labels 列表
data_shuffled = [data[i] for i in random_indices]
labels_shuffled = [labels[i] for i in random_indices]
raw_data_shuffled = [train_data[i] for i in random_indices]

# 以8：1：1的比例划分训练集、验证集、测试集
NumOfData = len(data_shuffled)
TrainNum = int(NumOfData * 0.8)
ValNum = int(NumOfData * 0.1)
TestNum = NumOfData - TrainNum - ValNum

# 要切换到的目标目录的路径
new_directory = "TrainCode/Data"  # 将此路径替换为您想要的目标目录

# 使用os.chdir()函数改变工作目录
os.chdir(new_directory)

np.save('TrueData30-12_train.npy',data_shuffled[:TrainNum])
np.save('PreData30-12_train.npy',raw_data_shuffled[:TrainNum])
np.save('LabelData30-12_train.npy',labels_shuffled[:TrainNum])

np.save('TrueData30-12_val.npy',data_shuffled[TrainNum:TrainNum + ValNum])
np.save('PreData30-12_val.npy',raw_data_shuffled[TrainNum:TrainNum + ValNum])
np.save('LabelData30-12_val.npy',labels_shuffled[TrainNum:TrainNum + ValNum])

np.save('TrueData30-12_test.npy',data_shuffled[TrainNum + ValNum:])
np.save('PreData30-12_test.npy',raw_data_shuffled[TrainNum + ValNum:])
np.save('LabelData30-12_test.npy',labels_shuffled[TrainNum + ValNum:])


print("train size is ",TrainNum)
print("val size is ", ValNum)
print("test size is ", TestNum)