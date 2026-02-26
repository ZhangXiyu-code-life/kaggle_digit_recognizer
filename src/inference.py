from pathlib import Path
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader

# 优先按项目结构导入（src.dataset / src.model）。
# 兼容直接运行 `python src/inference.py` 的场景。
try:
    from src.dataset import MNISTDataset
    from src.model import SimpleCNN
except ModuleNotFoundError:
    from dataset import MNISTDataset
    from model import SimpleCNN


def run_inference(
    test_csv_path: str,
    model_path: str,
    submission_path: str,
    batch_size: int = 256,
) -> pd.DataFrame:
    """
    加载测试集与训练好的模型权重，生成 Kaggle 提交文件。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) 读取测试集（显式使用 is_test=True，仅返回图像张量）
    test_dataset = MNISTDataset(csv_path=test_csv_path, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 2) 实例化模型并加载最佳权重，切换到 eval 模式
    model = SimpleCNN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3) 遍历测试集并收集预测结果（取 logit 的 argmax）
    predictions: List[int] = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            pred_labels = outputs.argmax(dim=1)
            predictions.extend(pred_labels.cpu().tolist())

    # 4) 构建提交表：ImageId 从 1 开始，Label 为预测结果
    submission_df = pd.DataFrame(
        {
            "ImageId": range(1, len(predictions) + 1),
            "Label": predictions,
        }
    )

    # 5) 保存为 experiments/submission.csv，不包含 index
    save_path = Path(submission_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(save_path, index=False)
    print(f"Submission file saved to: {save_path}")

    return submission_df


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    test_csv = (base_dir / "../data/test.csv").resolve()
    best_model = (base_dir / "../experiments/best_model.pth").resolve()
    submission_csv = (base_dir / "../experiments/submission.csv").resolve()

    run_inference(
        test_csv_path=str(test_csv),
        model_path=str(best_model),
        submission_path=str(submission_csv),
        batch_size=256,
    )
