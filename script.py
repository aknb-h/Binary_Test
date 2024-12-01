import argparse
import torch
import torch.nn as nn
import torch.optim as optim
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


# 学習と検証のロスと精度をグラフとして保存
def save_training_plots(train_losses, val_losses, val_accuracies, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    # ロスのプロット
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    print(f"Loss plot saved to {os.path.join(output_dir, 'loss_plot.png')}")

    # 精度のプロット
    plt.figure()
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
    print(f"Accuracy plot saved to {os.path.join(output_dir, 'accuracy_plot.png')}")


# 学習関数
def train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=10, save_path="model.pth"):
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # 学習
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x = x.unsqueeze(-1)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        # 検証
        model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.unsqueeze(-1)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == y).sum().item()
        val_loss /= len(val_loader)
        val_accuracy = correct / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    # テスト
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.unsqueeze(-1)
            outputs = model(x)
            loss = criterion(outputs, y)
            test_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == y).sum().item()
    test_loss /= len(test_loader)
    test_accuracy = correct / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # モデル保存
    save_model(model, save_path)

    # プロットを保存
    save_training_plots(train_losses, val_losses, val_accuracies)


# 推論関数
def predict(model, data, sequence_length):
    model.eval()
    x = data[-sequence_length:]
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, input_dim)
    with torch.no_grad():
        output = model(x)
        predicted = torch.argmax(output, dim=1).item()
    return "Increase" if predicted == 1 else "Decrease"


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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.mode == "train":
        if args.demo:
            data = generate_demo_data()
            print("Training using demo data...")
        else:
            data = load_data(args.file_path)
            print(f"Training using data from {args.file_path}...")
        
        # データ分割
        dataset = TimeSeriesDataset(data, sequence_length, delta)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, save_path=args.model_path)

    elif args.mode == "infer":
        load_model(model, args.model_path)
        if args.demo:
            data = generate_demo_data()
            print("Inferring using demo data...")
        else:
            data = load_data(args.file_path)
            print(f"Inferring using data from {args.file_path}...")

        prediction = predict(model, data, sequence_length)
        print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
