import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
# 可以跑一个resnet的baseline或者使用简单的ViT去进行一个分类

# 1. 数据集类
class LDCTDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths  # 图像路径列表
        self.labels = labels  # 标签列表（0-20的整数）
        self.transform = transform  # 数据增强

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_dicom(self.image_paths[idx])  # 加载DICOM图像
        image = normalize(image)  # 归一化到[0,1]

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image).unsqueeze(0)  # (1, H, W)

        label = self.labels[idx]
        return image, label

#
# # 2. 模型定义
# class EfficientNetIQA(nn.Module):
#     def __init__(self, num_classes=21):
#         super().__init__()
#         self.base = EfficientNet.from_pretrained('efficientnet-v2-l')
#         # 修改第一层：输入通道1 (灰度图)
#         self.base._conv_stem = nn.Conv2d(1, 24, kernel_size=3, stride=2, bias=False)
#         # 修改最后一层：输出21类
#         self.base._fc = nn.Linear(self.base._fc.in_features, num_classes)
#
#     def forward(self, x):
#         return self.base(x)


# 3. 数据增强
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
])

# 4. 训练循环
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EfficientNetIQA().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(100):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 验证集评估
    with torch.no_grad():
        acc = evaluate(model, val_loader)
    print(f"Epoch {epoch + 1}, Val Acc: {acc:.4f}")


# 5. 评估指标（需计算PLCC/SROCC/KROCC）
def calculate_correlations(pred_scores, true_scores):
    # pred_scores: 模型输出的连续分数（通过概率加权计算）
    # true_scores: 放射科医生平均评分
    plcc = np.corrcoef(pred_scores, true_scores)[0, 1]
    srocc = spearmanr(pred_scores, true_scores).correlation
    krocc = kendalltau(pred_scores, true_scores).correlation
    return plcc, srocc, krocc