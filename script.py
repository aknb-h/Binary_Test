import argparse
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import os
import matplotlib.pyplot as plt


# データセットクラス
class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length, delta):
        self.data = data
        self.sequence_length = sequence_length
        self.delta = delta

    def __len__(self):
        return len(self.data) - self.sequence_length - self.delta

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length + self.delta] - self.data[idx + self.sequence_length]
        label = 1 if y > 0 else 0
        return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# Transformerモデル
class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, hidden_dim, num_layers, output_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 500, embed_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout),
            num_layers
        )
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        out = self.fc(x.mean(dim=1))  # 平均プーリング
        return out

# データをCSVから読み込む関数
def load_data(file_path):
    df = pd.read_csv(file_path)
    data = df['value'].values
    return data


# デモ用データ生成関数
def generate_demo_data(length=1000):
    return np.sin(np.linspace(0, 100, length))


# モデルを保存する関数
def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# モデルをロードする関数
def load_model(model, path="model.pth"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file {path} not found.")
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")


# 推論関数（スコア付き）
def predict(model, data, sequence_length):
    model.eval()
    x = data[-sequence_length:]
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, input_dim)
    with torch.no_grad():
        output = model(x)  # ログits
        probabilities = torch.softmax(output, dim=1)  # ソフトマックスで確率を計算
        predicted = torch.argmax(probabilities, dim=1).item()
        score = probabilities[0, predicted].item()  # 信頼度スコア

    label = "Increase" if predicted == 1 else "Decrease"
    return label, score


# パラメータの数を計算する関数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# メイン処理
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "infer"], required=True,
                        help="Mode of operation: train or infer")
    parser.add_argument("--file_path", type=str, default="time_series_data.csv",
                        help="Path to the CSV file for training or inference")
    parser.add_argument("--model_path", type=str, default="model.pth",
                        help="Path to save/load the model")
    parser.add_argument("--demo", action="store_true", help="Use demo data instead of external files")
    args = parser.parse_args()

    # パラメータ設定
    sequence_length = 30
    delta = 5
    model = TransformerModel(input_dim=1, embed_dim=64, num_heads=4, hidden_dim=128, num_layers=2, output_dim=2)

    if args.mode == "infer":
        load_model(model, args.model_path)
        if args.demo:
            data = generate_demo_data()
            print("Inferring using demo data...")
        else:
            data = load_data(args.file_path)
            print(f"Inferring using data from {args.file_path}...")

        label, score = predict(model, data, sequence_length)
        print(f"Prediction: {label}, Score: {score:.4f}")


if __name__ == "__main__":
    main()
