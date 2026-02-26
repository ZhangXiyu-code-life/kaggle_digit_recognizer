import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class MNISTDataset(Dataset):
    def __init__(self, csv_path: str, is_test: bool = False):
        data = pd.read_csv(csv_path)
        self.is_test = is_test
        self.has_labels = (not is_test) and ("label" in data.columns)

        if self.has_labels:
            labels = data["label"].to_numpy(dtype="int64")
            features = data.drop(columns=["label"]).to_numpy(dtype="float32")
            self.labels = torch.from_numpy(labels).long()
        else:
            # 测试集或无标签数据：仅保留特征列。
            if "label" in data.columns:
                features = data.drop(columns=["label"]).to_numpy(dtype="float32")
            else:
                features = data.to_numpy(dtype="float32")
            self.labels = None

        features = features.reshape(-1, 1, 28, 28) / 255.0
        self.images = torch.from_numpy(features).float()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index]
        if self.has_labels:
            return image, self.labels[index]
        return image


def get_dataloaders(csv_path: str, batch_size: int, val_split: float):
    if not 0 < val_split < 1:
        raise ValueError("val_split 必须在 0 和 1 之间。")

    dataset = MNISTDataset(csv_path)
    if not dataset.has_labels:
        raise ValueError("输入的 CSV 不包含 label 列，无法划分训练/验证集。")

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    if val_size == 0 or train_size == 0:
        raise ValueError("val_split 导致训练集或验证集为空，请调整比例。")

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
