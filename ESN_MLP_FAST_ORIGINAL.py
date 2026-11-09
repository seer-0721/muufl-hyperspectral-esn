#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESN_MLP_FAST.py — NTU 自选21类（硬编码）+ 逐类准确率 + 诊断输出 + 每epoch指标导出 + 正/负EA曲线
- 硬编码 21 类：POSITIVE_ACTIONS + NEGATIVE_ACTIONS 的顺序 = 类别 0..20
- 扫描更健壮：递归 rglob，支持 .skeleton / .SKELETON，A(\d{3}) 优先，退化到 A(\d{2})
- 详细诊断：统计“总文件数/含A号文件数/命中21类数”，列举前若干不匹配样本
- 训练过程：每个 epoch 计算 train/val 指标并在训练结束后导出 metrics_epoch.csv
- EA曲线：按 epoch 导出正/负样本（基于 TARGET_ACTIONS 列表）在验证集上的准确率 ea_pos_neg.csv
- 训练结束：总体 acc、Top-k、逐类 acc，混淆矩阵CSV
"""

import os, re, warnings, argparse, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, top_k_accuracy_score
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- 1) 21类硬编码（顺序就是类别索引） --------------------
POSITIVE_ACTIONS = [5, 23, 28, 56, 53, 44, 45, 47, 48, 36]  # 10
NEGATIVE_ACTIONS = [9, 22, 35, 54, 57, 37, 30, 31, 41, 42, 40]  # 11
TARGET_ACTIONS = POSITIVE_ACTIONS + NEGATIVE_ACTIONS  # 21
OUT_DIM = len(TARGET_ACTIONS)
AID2CLS = {aid: i for i, aid in enumerate(TARGET_ACTIONS)}
CLS2AID = {v: k for k, v in AID2CLS.items()}
LABEL_NAMES = [f"A{aid:03d}" for aid in TARGET_ACTIONS]

# 正/负集合（按类别索引）
POS_CLS = [AID2CLS[aid] for aid in POSITIVE_ACTIONS]
NEG_CLS = [AID2CLS[aid] for aid in NEGATIVE_ACTIONS]


# -------------------- 2) 默认参数 --------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--skel-dir",
        type=str,
        default="./cleaned_21actions",
        help="根目录（递归搜索 *.skeleton）",
    )
    ap.add_argument("--cache-dir", type=str, default="./abc_cache_21actions")
    ap.add_argument("--max-frames", type=int, default=300)
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
    return ap.parse_args()


# -------------------- 3) 读 skeleton + 特征 --------------------
def parse_skeleton_file(path: Path, max_frames=None):
    with open(path, "r") as f:
        try:
            n_frames = int(f.readline())
        except:
            return None
        frames = []
        steps = n_frames if max_frames is None else min(n_frames, max_frames)
        for _ in range(steps):
            line = f.readline()
            if not line:
                break
            nb = int(line)
            if nb < 1:
                return None
            body0 = None
            for b in range(nb):
                f.readline()  # body meta
                n_joints = int(f.readline())
                if b == 0:
                    pts = [
                        list(map(float, f.readline().split()[:3]))
                        for _ in range(n_joints)
                    ]
                    body0 = np.asarray(pts, np.float32)
                else:
                    for _ in range(n_joints):
                        f.readline()
            frames.append(body0)
        if any(x is None for x in frames) or not frames:
            return None
        return np.stack(frames)  # (T,25,3)


def transform(coords, center_idx=1):
    center = coords[:, center_idx : center_idx + 1, :]
    rel = coords - center
    vel = np.zeros_like(coords, dtype=np.float32)
    vel[1:] = coords[1:] - coords[:-1]
    return np.hstack(
        [rel.reshape(len(rel), -1), vel.reshape(len(vel), -1)]
    )  # (T, 25*3*2)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=512, out_dim=OUT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# -------------------- 4) RFF + ESN --------------------
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
    phi = scale * np.cos(seq @ W_rff + b_rff)  # (T, rff_dim)
    states = esn_forward(phi, W_in, W, leak)  # (T, res_dim)
    return np.hstack(
        [states.mean(0), states.max(0), states.var(0), states[-1]]
    )  # (4*res_dim,)


# -------------------- 5) 文件名解析 + 扫描诊断 --------------------
def parse_action_id(name: str):
    # 优先 A(\d{3})，若没有再退化 A(\d{2})
    m = re.search(r"A(\d{3})", name)
    if not m:
        m = re.search(r"A(\d{2})\D", name)  # 少数数据可能写成 A56_...
    if not m:
        return None
    return int(m.group(1))


def scan_skeletons(root: Path):
    all_files = list(root.rglob("*.skeleton")) + list(root.rglob("*.SKELETON"))
    if not all_files:
        return [], {
            "total": 0,
            "has_a": 0,
            "in_21": 0,
            "samples_no_a": [],
            "samples_not_in_21": [],
        }
    has_a, in_21 = [], []
    samples_no_a, samples_not_in_21 = [], []
    for p in all_files:
        aid = parse_action_id(p.name)
        if aid is None:
            if len(samples_no_a) < 5:
                samples_no_a.append(p.name)
            continue
        has_a.append(p)
        if aid in AID2CLS:
            in_21.append(p)
        else:
            if len(samples_not_in_21) < 5:
                samples_not_in_21.append(p.name)
    stats = {
        "total": len(all_files),
        "has_a": len(has_a),
        "in_21": len(in_21),
        "samples_no_a": samples_no_a,
        "samples_not_in_21": samples_not_in_21,
    }
    return in_21, stats


# -------------------- 6) 指标计算与评估 --------------------
def per_class_accuracy(y_true, y_pred, n_cls=OUT_DIM):
    accs, support = np.zeros(n_cls), np.zeros(n_cls, dtype=np.int64)
    for c in range(n_cls):
        idx = y_true == c
        support[c] = int(idx.sum())
        accs[c] = (y_pred[idx] == c).mean() if support[c] > 0 else np.nan
    return accs, support


def subset_acc(y_true_np, y_pred_np, subset_cls):
    mask = np.isin(y_true_np, np.asarray(subset_cls))
    n = int(mask.sum())
    if n == 0:
        return np.nan, 0
    return float((y_pred_np[mask] == y_true_np[mask]).mean()), n


def evaluate(model, X, y, cache_dir: Path):
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
        for i, nm in enumerate(LABEL_NAMES):
            if np.isnan(accs[i]):
                print(f"  {nm}:  N/A   | n={support[i]}")
            else:
                print(f"  {nm}: {accs[i] * 100:6.2f}% | n={support[i]}")

        cm = confusion_matrix(y_true, y_pred, labels=list(range(OUT_DIM)))
        header = np.array([""] + LABEL_NAMES, dtype=object)
        cm_with_labels = np.empty((OUT_DIM + 1, OUT_DIM + 1), dtype=object)
        cm_with_labels[0, :] = header
        cm_with_labels[1:, 0] = LABEL_NAMES
        cm_with_labels[1:, 1:] = cm.astype(int)
        np.savetxt(
            cache_dir / "cm_val.csv",
            cm_with_labels,
            fmt="%s",
            delimiter=",",
        )
        print(f"[save] confusion matrix -> {cache_dir / 'cm_val.csv'}")
    return overall


# -------------------- 7) 主流程 --------------------
def main():
    args = get_args()
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    skel_root = Path(args.skel_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 打印映射
    print("[Mapping] class 0..20 ->", ", ".join(LABEL_NAMES))
    with open(cache_dir / "mapping.txt", "w") as f:
        for i, nm in enumerate(LABEL_NAMES):
            f.write(f"{i},{nm}\n")

    # 扫描
    if not skel_root.exists():
        raise FileNotFoundError(f"SKEL_DIR 不存在: {skel_root}")

    files, stats = scan_skeletons(skel_root)
    print(
        f"[Scan] total={stats['total']}, with_Axxx={stats['has_a']}, in_21={stats['in_21']}"
    )
    if stats["samples_no_a"]:
        print("  例：未能解析 Axxx 的文件名：", "; ".join(stats["samples_no_a"]))
    if stats["samples_not_in_21"]:
        print("  例：Axxx 不在 21 列表：", "; ".join(stats["samples_not_in_21"]))

    if not files:
        raise RuntimeError(
            "未找到任何属于 21 类目标动作的 .skeleton 文件；请检查 --skel-dir 路径或文件名中的 Axxx 是否匹配。"
        )

    # 解析 -> 特征
    seqs, labels = [], []
    for p in tqdm(files, desc="parse"):
        coords = parse_skeleton_file(p, args.max_frames)
        if coords is None:
            continue
        seqs.append(transform(coords))
        aid = parse_action_id(p.name)
        labels.append(AID2CLS[aid])

    if not seqs:
        raise RuntimeError("解析失败：没有有效序列（文件损坏或帧缺失）。")

    X = np.vstack(seqs).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    print(
        f"[Data] X={X.shape}, y={y.shape} | classes covered: {sorted(set(y.tolist()))}"
    )

    # 固定 RFF & ESN
    in_dim = X.shape[1]
    W_rff, b_rff, scale = build_random_features(rng, in_dim, args.rff_dim)
    W_in = rng.randn(args.res_dim, args.rff_dim).astype(np.float32)
    W = rng.randn(args.res_dim, args.res_dim).astype(np.float32)
    eigmax = float(np.abs(np.linalg.eigvals(W)).max())
    W *= np.float32(args.sr / (eigmax + 1e-12))

    # 编码
    feats = []
    for i in tqdm(range(len(seqs)), desc="encode"):
        feats.append(encode_sequence(seqs[i], W_rff, b_rff, scale, W_in, W, args.leak))
    Xf = torch.tensor(np.vstack(feats), dtype=torch.float32, device=args.device)
    y_all = torch.tensor(y, dtype=torch.long, device=args.device)

    # 划分
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

    # 模型
    model = MLP(in_dim=X_tr.shape[1], hidden=args.hidden, out_dim=OUT_DIM).to(
        args.device
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    # --------- 训练记录缓存（训练完一次性写CSV） ---------
    epoch_rows = []  # metrics_epoch.csv 行
    ea_rows = []  # ea_pos_neg.csv 行（正/负样本EA）

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

            # 正/负样本子集准确率（EA曲线点）
            pos_acc, pos_n = subset_acc(y_true_np, y_pred_np, POS_CLS)
            neg_acc, neg_n = subset_acc(y_true_np, y_pred_np, NEG_CLS)

        # 进度条展示
        pbar.set_postfix(
            tr_loss=f"{loss_tr.item():.4f}",
            tr_acc=f"{tr_acc * 100:.2f}%",
            va_loss=f"{loss_val.item():.4f}",
            va_acc=f"{val_acc * 100:.2f}%",
        )

        # 记录到缓存（稍后统一写CSV）
        epoch_rows.append(
            {
                "epoch": ep,
                "train_loss": float(loss_tr.item()),
                "train_acc": float(tr_acc),
                "val_loss": float(loss_val.item()),
                "val_acc": float(val_acc),
                "val_top3": float(val_top3),
                "pos_acc": float(pos_acc) if not np.isnan(pos_acc) else "",
                "pos_n": int(pos_n),
                "neg_acc": float(neg_acc) if not np.isnan(neg_acc) else "",
                "neg_n": int(neg_n),
            }
        )
        ea_rows.append(
            {
                "epoch": ep,
                "pos_acc": float(pos_acc) if not np.isnan(pos_acc) else "",
                "pos_n": int(pos_n),
                "neg_acc": float(neg_acc) if not np.isnan(neg_acc) else "",
                "neg_n": int(neg_n),
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
                "pos_acc",
                "pos_n",
                "neg_acc",
                "neg_n",
            ],
        )
        writer.writeheader()
        writer.writerows(epoch_rows)
    print(f"[save] per-epoch metrics -> {metrics_csv}")

    ea_csv = cache_dir / "ea_pos_neg.csv"
    with open(ea_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "pos_acc", "pos_n", "neg_acc", "neg_n"]
        )
        writer.writeheader()
        writer.writerows(ea_rows)
    print(f"[save] EA curve (pos/neg) -> {ea_csv}")

    # ---- 最终验证 + 模型/特征保存（保留原逻辑） ----
    overall = evaluate(model, X_val, y_val, cache_dir)
    torch.save(model.state_dict(), cache_dir / "mlp_model.pth")
    torch.save(
        {"features": Xf.cpu(), "labels": y_all.cpu()}, cache_dir / "features_labels.pt"
    )
    print(f"[save] model -> {cache_dir / 'mlp_model.pth'}")
    print(f"[done] Val Overall Acc = {overall * 100:.2f}%")


if __name__ == "__main__":
    main()
