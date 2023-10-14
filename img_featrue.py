import os
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import pandas as pd
from PIL import Image
import numpy as np

# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 定义卷积层、池化层和全连接层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 300 * 300, 16)  # 假设输入图片尺寸为 600x600

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 300 * 300)  # 假设输入图片尺寸为 600x600
        x = self.fc(x)
        return x

# 创建模型实例
model = CNN()

model.load_state_dict(torch.load('pic_model.pth'))
filenames = os.listdir('merged_folder')
new_df = pd.DataFrame(columns=['编号','特征'])

for i in range(0,4272):
    filename = filenames[i]
    image_path = os.path.join('merged_folder', filename)
    image = Image.open(image_path)
    image_tensor = ToTensor()(image)  # 转换为张量
    outputs = model(image_tensor)
    new_df = new_df.append({'编号':filename.replace(".jpg",""),'特征':outputs.detach().numpy()},ignore_index=True)

# new_df.to_excel("feature.xlsx", index=False)

# df = pd.read_excel('feature.xlsx')
# new_df = pd.DataFrame(columns = ['编号','特征'])
# for i in range(0,4272):
#     new_df = new_df.append({'编号':df.loc[i]["编号"],'特征':(np.array(df.loc[i]["特征"][1:-1]))},ignore_index=True)
# new_df.to_excel("feature_vector.xlsx", index=False)