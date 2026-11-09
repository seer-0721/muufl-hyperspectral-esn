#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESN_MLP_MUUFL.py — MUUFL Gulfport高光谱分类 + ESN + MLP
基于原始ESN_MLP_FAST.py迁移，适配MUUFL数据集

主要改动：
1. 数据加载：从.skeleton文件改为.mat文件
2. 特征提取：从骨架坐标改为光谱序列
3. 类别定义：从21个动作改为11个材料类别
4. 输入组织：像素级光谱序列
"""

import os, re, warnings, argparse, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, top_k_accuracy_score
from tqdm import tqdm

# 导入MUUFL工具
from muufl_utils import MUUFLDataset

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- 1) MUUFL类别定义 --------------------
# 使用MUUFLDataset中的类别映射
MUUF_CLASSES = 11  # 11个材料类别（不包括unlabeled）
OUT_DIM = MUUF_CLASSES


# -------------------- 2) 默认参数 --------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="数据目录（包含.mat文件）",
    )
    ap.add_argument(
        "--mat-file",
        type=str,
        default="muufl_gulfport_campus_1_hsi_220_label.mat",
        help=".mat文件名",
    )
    ap.add_argument("--cache-dir", type=str, default="./cache")
    ap.add_argument("--test-split", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    # 模型超参
    ap.add_argument("--rff-dim", type=int, default=1280)
    ap.add_argument("--res-dim", type=int, default=600)
    ap.add_argument("--sr", type=float, default=0.7526922197568455)  # spectral radius
    ap.add_argument("--leak", type=float, default=0.8316475567419054)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=30)
    # 数据预处理
    ap.add_argument("--normalize", action="store_true", help="是否归一化光谱数据")
    ap.add_argument("--use-cache", action="store_true", help="使用缓存的特征数据")
    return ap.parse_args()


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=512, out_dim=OUT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# -------------------- 3) RFF + ESN --------------------
def build_random_features(rng, in_dim, rff_dim):
    W = rng.randn(in_dim, rff_dim).astype(np.float32)
    b = rng.uniform(0, 2 * np.pi, size=rff_dim).astype(np.float32)
    scale = np.float32(np.sqrt(2.0 / rff_dim))
    return W, b, scale


def esn_forward(seq_phi, W_in, W, leak):
    res_dim = W.shape[0]
    state = np.zeros(res_dim, dtype=np.float32)
    T = seq_phi.shape[0]
    states = np.empty((T, res_dim), dtype=np.float32)
    for t in range(T):
        h = np.tanh(W_in @ seq_phi[t] + W @ state)
        state = (1.0 - leak) * state + leak * h
        states[t] = state
    return states


def encode_sequence(seq, W_rff, b_rff, scale, W_in, W, leak):
    """编码序列为ESN特征

    注意：对于光谱数据，每个像素是单时间步的64维向量
    序列长度T=1，但ESN仍能处理
    """
    phi = scale * np.cos(seq @ W_rff + b_rff)  # (T, rff_dim)
    states = esn_forward(phi, W_in, W, leak)  # (T, res_dim)

    # 对于单时间步，直接返回状态
    if states.shape[0] == 1:
        return states[0]  # (res_dim,)
    else:
        # 多时间步的聚合
        return np.hstack(
            [states.mean(0), states.max(0), states.var(0), states[-1]]
        )  # (4*res_dim,)


# -------------------- 4) 指标计算与评估 --------------------
def per_class_accuracy(y_true, y_pred, n_cls=OUT_DIM):
    accs, support = np.zeros(n_cls), np.zeros(n_cls, dtype=np.int64)
    for c in range(n_cls):
        idx = y_true == c
        support[c] = int(idx.sum())
        accs[c] = (y_pred[idx] == c).mean() if support[c] > 0 else np.nan
    return accs, support


def evaluate(model, X, y, cache_dir: Path, class_names):
    """最终验证 + 混淆矩阵导出"""
    model.eval()
    with torch.no_grad():
        logits = model(X)
        pred = logits.argmax(1)
        y_true = y.cpu().numpy()
        y_pred = pred.cpu().numpy()

        overall = accuracy_score(y_true, y_pred)
        print(f"[Val] Overall Acc = {overall * 100:.2f}%")

        try:
            top3 = top_k_accuracy_score(
                y_true,
                logits.cpu().numpy(),
                k=min(3, OUT_DIM),
                labels=list(range(OUT_DIM)),
            )
            print(f"[Val] Top-3 Acc    = {top3 * 100:.2f}%")
        except Exception:
            pass

        accs, support = per_class_accuracy(y_true, y_pred, OUT_DIM)
        print("[Val] Per-class Accuracies (class: acc% | support):")
        for i in range(OUT_DIM):
            class_name = class_names.get(i, f"class_{i}")
            if np.isnan(accs[i]):
                print(f"  {class_name}:  N/A   | n={support[i]}")
            else:
                print(f"  {class_name}: {accs[i] * 100:6.2f}% | n={support[i]}")

        cm = confusion_matrix(y_true, y_pred, labels=list(range(OUT_DIM)))
        # 创建类别名称列表
        cls_names = [class_names.get(i, f"class_{i}") for i in range(OUT_DIM)]
        header = np.array([""] + cls_names, dtype=object)
        cm_with_labels = np.empty((OUT_DIM + 1, OUT_DIM + 1), dtype=object)
        cm_with_labels[0, :] = header
        cm_with_labels[1:, 0] = cls_names
        cm_with_labels[1:, 1:] = cm.astype(int)
        np.savetxt(
            cache_dir / "cm_val.csv",
            cm_with_labels,
            fmt="%s",
            delimiter=",",
        )
        print(f"[save] confusion matrix -> {cache_dir / 'cm_val.csv'}")
    return overall


# -------------------- 5) 主流程 --------------------
def main():
    args = get_args()
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 打印配置
    print(f"[Config] 数据目录: {data_dir}")
    print(f"[Config] 设备: {args.device}")
    print(f"[Config] 类别数: {OUT_DIM}")

    # 创建类别映射文件
    muufl_dataset = MUUFLDataset(data_dir=data_dir)
    class_names = {i: muufl_dataset.CLASS_NAMES[i + 1] for i in range(OUT_DIM)}

    with open(cache_dir / "mapping.txt", "w") as f:
        for i in range(OUT_DIM):
            f.write(f"{i},{class_names[i]}\n")
    print(f"[save] 类别映射 -> {cache_dir / 'mapping.txt'}")

    # 检查缓存
    cache_file = cache_dir / "muufl_features.pt"
    if args.use_cache and cache_file.exists():
        print(f"[Cache] 从缓存加载特征: {cache_file}")
        cache_data = torch.load(cache_file)
        Xf = cache_data["features"].to(args.device)
        y_all = cache_data["labels"].to(args.device)
        print(f"[Cache] 特征: {Xf.shape}, 标签: {y_all.shape}")

    else:
        # 加载和预处理数据
        print("[Data] 加载MUUFL数据集...")
        mat_path = data_dir / args.mat_file

        if not mat_path.exists():
            raise FileNotFoundError(f"MAT文件不存在: {mat_path}")

        # 准备数据集
        spectra, labels = muufl_dataset.prepare_dataset(
            mat_path=mat_path, normalize=args.normalize, remap_labels=True
        )

        print(f"[Data] 原始光谱: {spectra.shape}, 标签: {labels.shape}")

        # 固定 RFF & ESN
        in_dim = spectra.shape[1]  # 波段数
        print(
            f"[ESN] 输入维度: {in_dim}, RFF维度: {args.rff_dim}, 储备池: {args.res_dim}"
        )

        W_rff, b_rff, scale = build_random_features(rng, in_dim, args.rff_dim)
        W_in = rng.randn(args.res_dim, args.rff_dim).astype(np.float32)
        W = rng.randn(args.res_dim, args.res_dim).astype(np.float32)
        eigmax = float(np.abs(np.linalg.eigvals(W)).max())
        W *= np.float32(args.sr / (eigmax + 1e-12))

        # 编码光谱为ESN特征
        print("[ESN] 编码光谱序列...")
        feats = []
        batch_size = 1000

        for i in tqdm(range(0, len(spectra), batch_size), desc="encode"):
            batch = spectra[i : i + batch_size]
            # 每个样本是单时间步序列
            for j in range(len(batch)):
                # 将光谱作为序列输入（序列长度=1）
                seq = batch[j : j + 1]  # (1, in_dim)
                feat = encode_sequence(seq, W_rff, b_rff, scale, W_in, W, args.leak)
                feats.append(feat)

        Xf = torch.tensor(np.vstack(feats), dtype=torch.float32, device=args.device)
        y_all = torch.tensor(labels, dtype=torch.long, device=args.device)

        print(f"[ESN] 编码后特征: {Xf.shape}")

        # 保存到缓存
        torch.save({"features": Xf.cpu(), "labels": y_all.cpu()}, cache_file)
        print(f"[save] 特征缓存 -> {cache_file}")

    # 划分训练/验证集
    print(f"[Split] 划分训练/验证集 (test_split={args.test_split})")
    X_tr, X_val, y_tr, y_val = train_test_split(
        Xf.cpu(),
        y_all.cpu(),
        test_size=args.test_split,
        stratify=y_all.cpu(),
        random_state=args.seed,
    )
    X_tr, X_val, y_tr, y_val = (
        X_tr.to(args.device),
        X_val.to(args.device),
        y_tr.to(args.device),
        y_val.to(args.device),
    )

    print(f"[Split] 训练集: {X_tr.shape}, 验证集: {X_val.shape}")

    # 模型
    model = MLP(in_dim=X_tr.shape[1], hidden=args.hidden, out_dim=OUT_DIM).to(
        args.device
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    print(f"[Model] MLP: {X_tr.shape[1]} -> {args.hidden} -> {OUT_DIM}")
    print(f"[Train] 学习率: {args.lr}, 轮数: {args.epochs}")

    # --------- 训练记录缓存 ---------
    epoch_rows = []  # metrics_epoch.csv 行

    # 训练
    pbar = tqdm(range(1, args.epochs + 1), desc="train", unit="epoch")
    for ep in pbar:
        # ---- Train ----
        model.train()
        opt.zero_grad()
        logits_tr = model(X_tr)
        loss_tr = crit(logits_tr, y_tr)
        loss_tr.backward()
        opt.step()

        with torch.no_grad():
            tr_acc = accuracy_score(
                y_tr.detach().cpu().numpy(), logits_tr.detach().argmax(1).cpu().numpy()
            )

        # ---- Val ----
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val)
            loss_val = crit(logits_val, y_val)
            y_true_np = y_val.detach().cpu().numpy()
            y_pred_np = logits_val.detach().argmax(1).cpu().numpy()
            val_acc = accuracy_score(y_true_np, y_pred_np)
            try:
                val_top3 = top_k_accuracy_score(
                    y_true_np,
                    logits_val.detach().cpu().numpy(),
                    k=min(3, OUT_DIM),
                    labels=list(range(OUT_DIM)),
                )
            except Exception:
                val_top3 = float("nan")

        # 进度条展示
        pbar.set_postfix(
            tr_loss=f"{loss_tr.item():.4f}",
            tr_acc=f"{tr_acc * 100:.2f}%",
            va_loss=f"{loss_val.item():.4f}",
            va_acc=f"{val_acc * 100:.2f}%",
        )

        # 记录到缓存
        epoch_rows.append(
            {
                "epoch": ep,
                "train_loss": float(loss_tr.item()),
                "train_acc": float(tr_acc),
                "val_loss": float(loss_val.item()),
                "val_acc": float(val_acc),
                "val_top3": float(val_top3),
            }
        )

    # --------- 训练完成后统一导出CSV ---------
    metrics_csv = cache_dir / "metrics_epoch.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "val_top3",
            ],
        )
        writer.writeheader()
        writer.writerows(epoch_rows)
    print(f"[save] per-epoch metrics -> {metrics_csv}")

    # ---- 最终验证 + 模型保存 ----
    overall = evaluate(model, X_val, y_val, cache_dir, class_names)
    torch.save(model.state_dict(), cache_dir / "mlp_model.pth")
    print(f"[save] model -> {cache_dir / 'mlp_model.pth'}")
    print(f"[done] Val Overall Acc = {overall * 100:.2f}%")


if __name__ == "__main__":
    main()

