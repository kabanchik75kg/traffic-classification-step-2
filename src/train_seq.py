#!/usr/bin/env python3
"""
Шаг 2.5 (sequence) — обучение sequence-моделей на φ_seq (2018).

Архитектуры берутся из реестра :mod:`seq_models` (cnn / tcn / bilstm / gru /
transformer) — все с единым интерфейсом ``forward(x, mask)``. Нормализация
size/IAT применяется на лету из ``splits.npz`` (μ,σ по train); dir не трогаем.
Обученная модель и пороги сохраняются для последующего переноса на 2017.

Для сопоставимости все sequence-модели обучаются одним протоколом (один и тот
же сплит, та же нормализация, та же ранняя остановка по val-PR-AUC, тот же
выбор порога по Нейману–Пирсону на валидации).

Запуск (venv активен):
    python train_seq.py --model cnn         --tensors ../tensors/2018 --splits ../splits/splits.npz
    python train_seq.py --model tcn         --tensors ../tensors/2018 --splits ../splits/splits.npz
    python train_seq.py --model bilstm      --tensors ../tensors/2018 --splits ../splits/splits.npz
    python train_seq.py --model gru         --tensors ../tensors/2018 --splits ../splits/splits.npz
    python train_seq.py --model transformer --tensors ../tensors/2018 --splits ../splits/splits.npz
Полезные флаги:
    --subsample-train 1000000   # на CPU рекомендуется (within-датасет насыщается)
    --seq-len 40                # абляция длины (срез из 60)
    --epochs 20 --batch 1024 --lr 1e-3

Зависит от: numpy, scikit-learn, torch, seq_models.
"""
import argparse
import glob
import os
import numpy as np


# ---------------- пороги/метрики Неймана–Пирсона (numpy, тестируется) ----------------
def threshold_at_fpr(benign_scores, p):
    """Наименьший τ, при котором доля benign со score>=τ не превышает p."""
    return float(np.quantile(benign_scores, 1.0 - p))


def threshold_at_tpr(attack_scores, tpr):
    """τ, при котором полнота на атаках = tpr."""
    return float(np.quantile(attack_scores, 1.0 - tpr))


def rates(scores, y, tau):
    """Вернуть (recall/TPR, FPR, precision) при пороге τ."""
    pred = scores >= tau
    P = y == 1; Nn = y == 0
    tpr = float((pred & P).sum() / max(P.sum(), 1))
    fpr = float((pred & Nn).sum() / max(Nn.sum(), 1))
    prec = float((pred & P).sum() / max(pred.sum(), 1))
    return tpr, fpr, prec


def per_class_recall(scores, y, gid, classes, tau):
    """Полнота по каждому классу атак при пороге τ: список (класс, recall, n)."""
    out = []
    for ci, c in enumerate(classes):
        m = (gid == ci) & (y == 1)
        n = int(m.sum())
        if n:
            out.append((c, float((scores[m] >= tau).mean()), n))
    return out


# ---------------- сборка + нормализация на лету (numpy, тестируется) ----------------
def load_and_prepare(tensors_dir, splits_path, seq_len=60):
    """Собрать тензоры всех дней, нормализовать size/IAT по μ,σ из splits.npz.

    Возвращает X16 (float16, [N,seq_len,3]), маску, метку y, разметку split
    (0/1/2) и глобальный id класса gid, а также список имён классов.
    Нормализация: ``(x-μ)/σ * mask`` (паддинг обнуляется), direction не трогаем.
    """
    z = np.load(splits_path, allow_pickle=False)
    classes = [str(c) for c in z["classes"]]
    mu_s, sg_s = float(z["mu_size"]), float(z["sigma_size"])
    mu_i, sg_i = float(z["mu_iat"]), float(z["sigma_iat"])

    keys = set(z.files)
    files, days = [], []
    for f in sorted(glob.glob(os.path.join(tensors_dir, "*.npz"))):
        d = os.path.splitext(os.path.basename(f))[0]
        if f"split_{d}" in keys:           # пропускаем посторонние .npz (напр. сам splits.npz)
            files.append(f); days.append(d)
    if not files:
        raise SystemExit("Не найдено дней с соответствующим split-ключом в splits.npz")
    counts = [len(z[f"split_{d}"]) for d in days]
    total = sum(counts)

    X16 = np.empty((total, seq_len, 3), dtype=np.float16)
    M = np.empty((total, seq_len), dtype=np.uint8)
    y = np.empty(total, dtype=np.int8)
    split = np.empty(total, dtype=np.int8)
    gid = np.empty(total, dtype=np.int16)

    off = 0
    for f, d, n in zip(files, days, counts):
        t = np.load(f, allow_pickle=False)
        X = t["X"][:, :seq_len, :].astype(np.float32)
        mask = t["mask"][:, :seq_len]
        # нормализация на лету; паддинг обнуляем маской
        X[..., 0] = (X[..., 0] - mu_s) / sg_s * mask
        X[..., 2] = (X[..., 2] - mu_i) / sg_i * mask
        X16[off:off + n] = X.astype(np.float16)
        M[off:off + n] = mask
        y[off:off + n] = t["y"]
        split[off:off + n] = z[f"split_{d}"]
        gid[off:off + n] = z[f"class_{d}"]
        off += n
    return X16, M, y, split, gid, classes


def main():
    import seq_models  # реестр архитектур (импортирует torch)

    ap = argparse.ArgumentParser(description="Обучение sequence-модели на φ_seq (2018)")
    ap.add_argument("--model", choices=seq_models.MODEL_NAMES, required=True)
    ap.add_argument("--tensors", default="../tensors/2018")
    ap.add_argument("--splits", default="../splits/splits.npz")
    ap.add_argument("--out", default="../models")
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--subsample-train", type=int, default=0, help="0=весь train")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import roc_auc_score, average_precision_score

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {dev}")
    if args.model in ("bilstm", "gru") and dev.type == "cpu":
        print("  ВНИМАНИЕ: рекуррентная модель на CPU обучается медленно. "
              "При долгой эпохе используй --subsample-train 1000000.")

    print("Загрузка тензоров + нормализация на лету...")
    X16, M, y, split, gid, classes = load_and_prepare(args.tensors, args.splits, args.seq_len)
    idx = {0: np.where(split == 0)[0], 1: np.where(split == 1)[0], 2: np.where(split == 2)[0]}
    if args.subsample_train and len(idx[0]) > args.subsample_train:
        rng = np.random.default_rng(args.seed)
        idx[0] = rng.choice(idx[0], args.subsample_train, replace=False)
    print(f"  train {len(idx[0]):,} | val {len(idx[1]):,} | test {len(idx[2]):,} | L={args.seq_len}")

    class DS(Dataset):
        def __init__(self, ids): self.ids = ids
        def __len__(self): return len(self.ids)
        def __getitem__(self, i):
            j = self.ids[i]
            return (torch.from_numpy(X16[j].astype(np.float32)),
                    torch.from_numpy(M[j].astype(np.float32)),
                    np.float32(y[j]))

    dl = lambda s, sh: DataLoader(DS(idx[s]), batch_size=args.batch, shuffle=sh, num_workers=0)
    tr_dl, va_dl, te_dl = dl(0, True), dl(1, False), dl(2, False)

    model = seq_models.make_model(args.model, args.seq_len).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = torch.nn.BCEWithLogitsLoss()

    def scores_for(loader):
        model.eval(); out = []
        with torch.no_grad():
            for xb, mb, _ in loader:
                logit = model(xb.to(dev), mb.to(dev))
                out.append(torch.sigmoid(logit).cpu().numpy())
        return np.concatenate(out)

    best_ap, best_state, bad = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, mb, yb in tr_dl:
            opt.zero_grad()
            loss = lossf(model(xb.to(dev), mb.to(dev)), yb.to(dev))
            loss.backward(); opt.step()
        ap_val = average_precision_score(y[idx[1]], scores_for(va_dl))
        print(f"  эпоха {ep:2d}: val PR-AUC = {ap_val:.5f}")
        if ap_val > best_ap + 1e-5:
            best_ap, best_state, bad = ap_val, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  ранняя остановка (нет улучшения {args.patience} эпох)")
                break

    model.load_state_dict(best_state)
    s_va, s_te = scores_for(va_dl), scores_for(te_dl)
    yv, yt = y[idx[1]].astype(int), y[idx[2]].astype(int)
    gt = gid[idx[2]]

    pts = {
        "FPR<=1%": threshold_at_fpr(s_va[yv == 0], 0.01),
        "FPR<=0.1%": threshold_at_fpr(s_va[yv == 0], 0.001),
        "TPR=95% (как этап1)": threshold_at_tpr(s_va[yv == 1], 0.95),
    }
    print(f"\n=== РЕЗУЛЬТАТЫ (φ_seq, {args.model.upper()}, L={args.seq_len}) ===")
    print(f"  ROC-AUC(test) = {roc_auc_score(yt, s_te):.4f}   PR-AUC(test) = {average_precision_score(yt, s_te):.4f}")
    print(f"  {'точка':<22} {'τ':>7} {'Recall':>8} {'FPR':>9} {'Precision':>10}")
    for name, tau in pts.items():
        tpr, fpr, prec = rates(s_te, yt, tau)
        print(f"  {name:<22} {tau:>7.4f} {tpr:>8.4f} {fpr:>9.5f} {prec:>10.4f}")

    tau = pts["FPR<=0.1%"]  # единая рабочая точка для per-class (как у seq и baseline)
    print(f"\n  Полнота по классам на test (порог FPR<=0.1%, τ={tau:.4f}):")
    for c, rec, n in per_class_recall(s_te, yt, gt, classes, tau):
        print(f"     {c:<20} recall={rec:.4f}  (n={n:,})")

    os.makedirs(args.out, exist_ok=True)
    torch.save(best_state, os.path.join(args.out, f"seq_{args.model}.pt"))
    np.savez(os.path.join(args.out, f"seq_{args.model}_meta.npz"),
             model=np.array(args.model), seq_len=np.array(args.seq_len),
             classes=np.array(classes, dtype="U40"),
             tau_fpr01=np.array(pts["FPR<=1%"]), tau_fpr001=np.array(pts["FPR<=0.1%"]),
             tau_tpr95=np.array(pts["TPR=95% (как этап1)"]))
    print(f"\n-> сохранено: {args.out}/seq_{args.model}.pt (+ meta). Эти веса и пороги пойдут на 2017.")


if __name__ == "__main__":
    main()
