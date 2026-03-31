import os
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

# ==========================================
# 1. 設定區塊
# ==========================================
MODEL_PATH = 'best_resnet101_standard_pro_v2.pth'
TEST_DIR = './data/test'
TRAIN_DIR = './data/train'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 模型結構 (須與訓練時完全一致)
# ==========================================


def get_model(num_classes=100):
    model = models.resnet101()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

# ==========================================
# 3. TTA 特用數據讀取工具
# ==========================================


class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(test_dir) if f.endswith(
                ('.png', '.jpg', '.jpeg'))]

    def __len__(self): return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(
            os.path.join(
                self.test_dir,
                img_name)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, os.path.splitext(img_name)[0]


# 🌟 TTA 專用轉換：這裡我們用 TenCrop (五張圖：四角+中心)
tta_transforms = transforms.Compose([
    # 1. Resize 比例建議為 1.05~1.15 倍。
    # 480 * 1.07 = 512，這是一個很穩定的比例。
    # 增加 interpolation=InterpolationMode.BICUBIC 可以獲得更細膩的縮放品質。
    transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),

    # 2. TenCrop 保持 480，確保模型看到的特徵大小與訓練時 (RandomResizedCrop 480) 一致
    transforms.TenCrop(480),

    # 3. 將 10 張圖轉為 Tensor 並標準化
    transforms.Lambda(lambda crops: torch.stack([transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(crop) for crop in crops]))
])

train_classes = datasets.ImageFolder(root=TRAIN_DIR).classes

# ==========================================
# 4. 執行 TTA 預測邏輯
# ==========================================


def run_tta_prediction():
    model = get_model(num_classes=100)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        model = model.to(DEVICE).eval()
        print("✅ 成功載入權重並啟動 TTA 預測模式")
    else:
        print(f"❌ 找不到權重檔 {MODEL_PATH}")
        return

    # batch_size 建議設小一點，因為 TenCrop 會讓資料量瞬間變 10 倍
    test_loader = DataLoader(
        TestDataset(
            TEST_DIR,
            tta_transforms),
        batch_size=16,
        shuffle=False,
        num_workers=4)

    results = []
    with torch.no_grad():
        for inputs, names in tqdm(test_loader, desc="TTA Predicting"):
            # inputs 的形狀會是 [batch, 10, 3, 224, 224]
            bs, n_crops, c, h, w = inputs.size()

            # 將 crops 展開進行預測
            input_combined = inputs.view(-1, c, h, w).to(DEVICE)
            outputs = model(input_combined)

            # 取得機率 (Softmax)
            probs = F.softmax(outputs, dim=1)

            # 將 10 張圖的機率平均
            avg_probs = probs.view(bs, n_crops, -1).mean(1)

            # 取得最終預測
            _, preds = torch.max(avg_probs, 1)

            for name, p in zip(names, preds.cpu().numpy()):
                results.append(
                    {"image_name": name, "pred_label": train_classes[p]})

    # 存檔
    df = pd.DataFrame(results).sort_values('image_name')
    output_name = 'submission_standard_tta.csv'
    df.to_csv(output_name, index=False)
    print(f"🎉 TTA 預測完成！結果已存至: {output_name}")


if __name__ == "__main__":
    run_tta_prediction()
