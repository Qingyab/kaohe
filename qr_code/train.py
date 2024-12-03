import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


# 自定义数据集类以加载图像和标签
class CustomDataset(Dataset):
    def __init__(self, json_dir, transform=None):
        self.json_dir = json_dir
        self.json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        self.transform = transform

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_file = os.path.join(self.json_dir, self.json_files[idx])
        with open(json_file, 'r') as f:
            data = json.load(f)

        image = Image.open(data['path']).convert('RGB')
        label = 0 if data['outputs']['object'][0]['name'] == 'q1' else 1

        if self.transform:
            image = self.transform(image)

        return image, label


# 定义 CNN 模型（使用预训练的 ResNet18 作为示例）
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # 2 分类问题

    def forward(self, x):
        return self.model(x)


def train_model(model, dataloader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')


# 数据变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 根据模型要求调整图像大小
    transforms.ToTensor(),  # 转换为 Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
])

# 加载数据
json_dir = 'C:/Users/22132/Desktop/pic/outputs'
dataset = CustomDataset(json_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 初始化模型、损失函数和优化器
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train_model(model, dataloader, criterion, optimizer, num_epochs=5)

# 保存模型为 ONNX 格式
dummy_input = torch.randn(1, 3, 224, 224)  # 为 ONNX 导出创建一个虚拟输入
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

print("Model exported as ONNX format.")
