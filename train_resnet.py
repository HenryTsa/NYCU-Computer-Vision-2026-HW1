import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 數據增強：RandAugment (現代標配)
# ==========================================
train_transforms = transforms.Compose([
    transforms.RandAugment(num_ops=2, magnitude=9),  # 🌟 自動組合多種增強
    transforms.RandomResizedCrop(480),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(480),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 2. 輔助函數：Cutmix 區域計算
# ==========================================


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

# ==========================================
# 3. 建立模型 (ResNet-101)
# ==========================================


def get_model(num_classes=100):
    # 🌟 載入官方預訓練權重
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    # 簡單強化分類頭，加入 Dropout 防止過擬合
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_ftrs, num_classes)
    )
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=100).to(device)

# ==========================================
# 4. 專業配置：AdamW + Label Smoothing
# ==========================================
# AdamW 比起傳統 Adam 有更好的權重衰減處理
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
# 🌟 Label Smoothing: 防止模型過於自信
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

num_epochs = 30
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

history = {'train_loss': [], 'val_acc': []}
best_acc = 0.0
os.makedirs('./plots', exist_ok=True)

# ==========================================
# 5. Training Loop (延遲開啟 Mixup/Cutmix)
# ==========================================
for epoch in range(num_epochs):
    model.train()
    running_loss, total_samples = 0.0, 0

    # 🌟 關鍵策略：前 2 輪關閉混合增強，第 3 輪 (index 2) 再開啟
    use_mix_cut = True if epoch >= 2 else False

    train_loader = DataLoader(
        datasets.ImageFolder('./data/train', train_transforms),
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        if use_mix_cut and np.random.rand() < 0.5:
            # 🎲 50% 機率進入 Mixup 或 Cutmix
            lam = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(inputs.size()[0]).to(device)
            target_a, target_b = targets, targets[rand_index]

            if np.random.rand() < 0.5:  # Mixup
                inputs = lam * inputs + (1 - lam) * inputs[rand_index]
            else:  # Cutmix
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[:,
                       :,
                       bbx1:bbx2,
                       bby1:bby2] = inputs[rand_index,
                                           :,
                                           bbx1:bbx2,
                                           bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                           (inputs.size()[-1] * inputs.size()[-2]))

            outputs = model(inputs)
            loss = lam * criterion(outputs, target_a) + \
                (1. - lam) * criterion(outputs, target_b)
        else:
            # 正常訓練 (前兩輪或隨機 50% 情況)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    # 驗證
    model.eval()
    val_correct, val_total = 0, 0
    val_loader = DataLoader(datasets.ImageFolder('./data/val', val_transforms),
                            batch_size=32, shuffle=False, num_workers=4)
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs.to(device))
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets.to(device)).sum().item()

    current_acc = 100. * val_correct / val_total
    history['train_loss'].append(running_loss / total_samples)
    history['val_acc'].append(current_acc)
    scheduler.step()

    print(f"📊 Val Acc: {current_acc:.2f}% | Best: {best_acc:.2f}%")
    if current_acc > best_acc:
        best_acc = current_acc
        torch.save(model.state_dict(), 'best_resnet101_standard_pro_v2.pth')

# ==========================================
# 6. 產出訓練曲線
# ==========================================
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'])
plt.title('Loss')
plt.subplot(1, 2, 2)
plt.plot(history['val_acc'])
plt.title('Accuracy')
plt.savefig('./plots/standard_pro_curve.png')
print("🏁 訓練完成！曲線圖已存至 ./plots/standard_pro_curve_2.png")
