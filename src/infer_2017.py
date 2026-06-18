#!/usr/bin/env python3
"""infer_2017.py — перенос моделей 2018 на тензоры 2017 (бинарно, без дообучения).
Нормализация size/iat — теми же mu/sigma из splits.npz (train-2018), 1:1 как train_seq.
Модель и forward — из train_seq.make_model. Тензоры 2017 не модифицируются.
Запуск:
  python infer_2017.py --model cnn
  python infer_2017.py --model bilstm
"""
import argparse, glob, os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from train_seq import make_model, threshold_at_fpr, rates   # твой код 1:1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "bilstm"], required=True)
    ap.add_argument("--tensors", default=os.path.expanduser("~/vkr_results_2017/tensors_2017"))
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--splits", default="splits/splits.npz")
    ap.add_argument("--out", default=os.path.expanduser("~/vkr_results_2017/scores_2017"))
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(args.splits, allow_pickle=False)
    mu_s, sg_s = float(z["mu_size"]), float(z["sigma_size"])
    mu_i, sg_i = float(z["mu_iat"]),  float(z["sigma_iat"])
    meta = np.load(os.path.join(args.models_dir, f"seq_{args.model}_meta.npz"), allow_pickle=True)
    seq_len = int(meta["seq_len"])
    tau01, tau001 = float(meta["tau_fpr01"]), float(meta["tau_fpr001"])

    model = make_model(args.model, seq_len).to(dev)
    model.load_state_dict(torch.load(os.path.join(args.models_dir, f"seq_{args.model}.pt"), map_location=dev))
    model.eval()
    print(f"{args.model.upper()} | seq_len={seq_len} | dev={dev} | μσ(size)={mu_s:.2f}/{sg_s:.2f} τ(0.1%)={tau001:.4f}")

    os.makedirs(args.out, exist_ok=True)
    S, Y, NAT = [], [], []
    for f in sorted(glob.glob(os.path.join(args.tensors, "*.npz"))):
        day = os.path.splitext(os.path.basename(f))[0]
        t = np.load(f, allow_pickle=False)
        X = t["X"][:, :seq_len, :].astype(np.float32)
        mask = t["mask"][:, :seq_len].astype(np.float32)
        X[..., 0] = (X[..., 0] - mu_s) / sg_s * mask      # ровно как train_seq
        X[..., 2] = (X[..., 2] - mu_i) / sg_i * mask
        y = t["y"].astype(int); nat = t["native_label"]
        s = []
        with torch.no_grad():
            for i in range(0, len(X), args.batch):
                xb = torch.from_numpy(X[i:i+args.batch]).to(dev)
                mb = torch.from_numpy(mask[i:i+args.batch]).to(dev)
                s.append(torch.sigmoid(model(xb, mb)).cpu().numpy())
        s = np.concatenate(s)
        np.savez_compressed(os.path.join(args.out, f"{args.model}_{day}.npz"), score=s, y=y, native_label=nat)
        auc = roc_auc_score(y, s) if (y.any() and (y == 0).any()) else float("nan")
        pr  = average_precision_score(y, s) if y.any() else float("nan")
        print(f"  {day:10} n={len(y):>7,} атак={int(y.sum()):>7,} | ROC-AUC={auc:.4f} PR-AUC={pr:.4f}")
        S.append(s); Y.append(y); NAT.append(nat)

    s = np.concatenate(S); y = np.concatenate(Y); nat = np.concatenate(NAT)
    print(f"\n=== ВЕСЬ 2017 ({args.model.upper()}) | потоков {len(y):,} | атак {int(y.sum()):,} ({100*y.mean():.1f}%) ===")
    print(f"  ROC-AUC = {roc_auc_score(y, s):.4f}   PR-AUC = {average_precision_score(y, s):.4f}")
    print(f"  {'точка (τ с 2018-val)':<20} {'τ':>8} {'Recall':>8} {'FPR':>9} {'Precision':>10}")
    for name, tau in [("FPR<=1%", tau01), ("FPR<=0.1%", tau001)]:
        tpr, fpr, prec = rates(s, y, tau)
        print(f"  {name:<20} {tau:>8.4f} {tpr:>8.4f} {fpr:>9.5f} {prec:>10.4f}")
    low = np.char.lower(nat.astype(str))
    for cls in ["portscan", "heartbleed"]:
        m = low == cls
        if m.any():
            print(f"  [unseen] {cls:<11} recall@FPR0.1% = {(s[m] >= tau001).mean():.4f} (n={int(m.sum()):,})")
    np.savez_compressed(os.path.join(args.out, f"{args.model}_ALL.npz"), score=s, y=y, native_label=nat)
    print(f"  -> {args.out}/")

if __name__ == "__main__":
    main()