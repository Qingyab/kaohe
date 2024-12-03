import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        try:
            with open(json_file, 'r',encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Failed to load {json_file}: {e}")
            return None, None  # 返回 None 以跳过错误的文件

        try:
            image = Image.open(data['path']).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to open image {data['path']}: {e}")
            return None, None  # 如果图像无法打开，返回 None

        label = 0 if data['outputs']['object'][0]['name'] == 'q1' else 1

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                logger.warning(f"Failed to transform image {data['path']}: {e}")
                return None, None  # 如果转换失败，返回 None

        return image, label


# 定义 CNN 模型（使用预训练的 ResNet18 作为示例）
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1  # 使用默认权重
        self.model = models.resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # 2 分类问题

    def forward(self, x):
        return self.model(x)


def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return None, None  # 如果批次为空，返回 None
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return images, labels


def train_model(model, dataloader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            if inputs is None or labels is None:
                continue  # 跳过空批次
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logger.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')


# 数据变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 根据模型要求调整图像大小
    transforms.ToTensor(),  # 转换为 Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
])

# 加载数据
json_dir = 'C:/Users/22132/Desktop/pic/outputs'
dataset = CustomDataset(json_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# 初始化模型、损失函数和优化器
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
try:
    train_model(model, dataloader, criterion, optimizer, num_epochs=5)
except Exception as e:
    logger.error(f"Training failed: {e}")

# 保存模型为 ONNX 格式
model.eval()  # 确保模型处于评估模式
dummy_input = torch.randn(1, 3, 224, 224)  # 为 ONNX 导出创建一个虚拟输入
try:
    torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    logger.info("Model exported as ONNX format.")
except Exception as e:
    logger.error(f"Failed to export model to ONNX: {e}")