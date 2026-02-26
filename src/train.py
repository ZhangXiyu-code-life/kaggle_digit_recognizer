from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# 优先按项目结构导入（src.dataset / src.model）。
# 兼容直接运行 `python src/train.py` 的场景，避免因导入路径差异报错。
try:
    from src.dataset import get_dataloaders
    from src.model import SimpleCNN
except ModuleNotFoundError:
    from dataset import get_dataloaders
    from model import SimpleCNN


def train_model(
    csv_path: str,
    batch_size: int = 64,
    val_split: float = 0.2,
    epochs: int = 5,
    model_save_path: str = "../experiments/best_model.pth",
) -> Tuple[SimpleCNN, float]:
    """
    训练 SimpleCNN 并返回训练后的模型与最佳验证集准确率。
    """
    # 自动选择计算设备：
    # - 若检测到 CUDA 可用，使用 GPU；
    # - 否则自动回退到 CPU（Windows 无 NVIDIA 显卡时也可正常运行）。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 获取训练/验证数据加载器。
    train_loader, val_loader = get_dataloaders(
        csv_path=csv_path,
        batch_size=batch_size,
        val_split=val_split,
    )

    # 初始化模型、损失函数与优化器。
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 准备模型保存路径（若目录不存在则自动创建）。
    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # =========================
        # 1) 训练阶段 (train mode)
        # =========================
        model.train()
        running_loss = 0.0
        total_train_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 清空上一轮梯度，前向传播，计算损失，反向传播并更新参数。
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 按样本数累加，便于计算整个 epoch 的平均训练损失。
            batch_size_current = images.size(0)
            running_loss += loss.item() * batch_size_current
            total_train_samples += batch_size_current

        train_loss = running_loss / total_train_samples if total_train_samples > 0 else 0.0

        # ============================
        # 2) 验证阶段 (eval mode)
        # ============================
        model.eval()
        correct = 0
        total_val_samples = 0

        # 验证阶段不需要梯度，节省显存并提升推理速度。
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total_val_samples += labels.size(0)

        val_acc = correct / total_val_samples if total_val_samples > 0 else 0.0

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Accuracy: {val_acc:.4%}"
        )

        # 仅当当前验证准确率更优时，保存最佳模型权重。
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Best model updated and saved to: {save_path}")

    print(f"Training completed. Best Val Accuracy: {best_val_acc:.4%}")
    return model, best_val_acc


if __name__ == "__main__":
    # 以当前脚本路径为基准构建相对路径，避免受运行目录影响。
    base_dir = Path(__file__).resolve().parent
    csv_path = (base_dir / "../data/train.csv").resolve()
    best_model_path = (base_dir / "../experiments/best_model.pth").resolve()

    train_model(
        csv_path=str(csv_path),
        batch_size=64,
        val_split=0.2,
        epochs=5,
        model_save_path=str(best_model_path),
    )
