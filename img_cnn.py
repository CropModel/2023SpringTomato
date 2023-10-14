import os
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import pandas as pd
from PIL import Image

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

# 定义训练数据、标签等
train_data = []
train_labels = []

# 图片文件夹路径
folder_path = 'merged_folder'
df = pd.read_excel('num_high.xlsx')
df.set_index('编号', inplace=True)
# 根据编号索引到高度

# 定义批处理大小和迭代次数
batch_size = 16
num_epochs = 20

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
total_samples = 4271
total_batches = total_samples // batch_size

for epoch in range(num_epochs):
    for batch in range(total_batches):
        # 获取当前批次的数据和标签
        train_data = []
        train_labels = []
        start_index = batch * batch_size
        end_index = start_index + batch_size
        # print(start_index,end_index,os.listdir(folder_path)[start_index:end_index])
        for filename in os.listdir(folder_path)[start_index:end_index]:
            # print(filename)
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            image_tensor = ToTensor()(image)  # 转换为张量
            train_data.append(image_tensor)
            label = height = df.loc[filename.replace(".jpg",''), '高度']
            train_labels.append([label])
        train_data = torch.stack(train_data)
        # train_labels = torch.tensor(train_labels)
        batch_data = train_data
        batch_labels = torch.Tensor(train_labels)

        # 前向传播
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 输出训练信息
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), 'pic_model2.pth')
