#!/usr/bin/env python3
"""
Шаг 2.5 (baseline) — XGBoost на агрегатах φ_agg, выведенных из НАШЕЙ же
последовательности (те же потоки и тот же сплит, что у seq-моделей).

Назначение: контролируемый полюс «агрегаты vs последовательность». Поскольку
данные, разбиение и разметка идентичны seq-моделям, разница в качестве переноса
на 2017 объясняется ТОЛЬКО представлением (13 агрегатов против попакетной
матрицы N×3). Это и есть честное доказательство тезиса работы — в отличие от
сравнения с агрегатами этапа 1, где менялись ещё и данные, и конвейер.

Модель и пороги сохраняются (как у seq), чтобы перенести baseline на 2017
скриптом infer_2017_baseline.py.

Запуск:
    python train_baseline.py --tensors ../tensors/2018 --splits ../splits/splits.npz --out ../models

Зависит от: numpy, xgboost, scikit-learn, metrics_np, train_seq.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from metrics_np import best_f1_threshold, best_mcc, metric_block
from train_seq import rates, threshold_at_fpr, threshold_at_tpr


# ---------------- агрегаты из последовательности (ядро, тестируется) ----------------
def aggregate_features(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """X [n,N,3]=[size,dir,iat], mask [n,N] -> φ_agg [n, 13].

    13 агрегатов потока: длина; mean/std/min/max/sum размера; mean/std/max IAT;
    доля пакетов «к жертве»; число смен направления; размер первого и последнего
    пакета. Считаются по реальным пакетам (через маску).
    """
    size = X[..., 0].astype(np.float64)
    direction = X[..., 1]
    iat = X[..., 2].astype(np.float64)
    m = mask.astype(np.float64)
    L = m.sum(1)                                   # длина потока (реальных пакетов)
    Ls = np.clip(L, 1, None)

    size_sum = (size * m).sum(1)
    size_mean = size_sum / Ls
    size_std = np.sqrt(np.clip((size * size * m).sum(1) / Ls - size_mean ** 2, 0, None))
    size_max = size.max(1)                          # паддинг=0, реальные>0
    size_min = np.where(mask == 1, size, np.inf).min(1)

    iat_mean = (iat * m).sum(1) / Ls
    iat_std = np.sqrt(np.clip((iat * iat * m).sum(1) / Ls - iat_mean ** 2, 0, None))
    iat_max = iat.max(1)

    frac_fwd = ((direction == 1) * m).sum(1) / Ls   # доля пакетов к жертве
    changes = (((direction[:, 1:] != direction[:, :-1]) & (mask[:, 1:] == 1)).sum(1)).astype(np.float64)

    first_size = size[:, 0]
    rows = np.arange(len(size))
    last_size = size[rows, (L.astype(int) - 1)]

    return np.column_stack([
        L, size_mean, size_std, size_min, size_max, size_sum,
        iat_mean, iat_std, iat_max, frac_fwd, changes, first_size, last_size,
    ]).astype(np.float32)


FEATURE_NAMES = ["len", "size_mean", "size_std", "size_min", "size_max", "size_sum",
                 "iat_mean", "iat_std", "iat_max", "frac_fwd", "dir_changes",
                 "first_size", "last_size"]


def build(tensors_dir: str, splits_path: str):
    """Собрать φ_agg всех дней + метку, split (0/1/2) и глобальный id класса."""
    z = np.load(splits_path, allow_pickle=False)
    classes = [str(c) for c in z["classes"]]
    keys = set(z.files)
    Xa, ys, sp, gid = [], [], [], []
    for f in sorted(glob.glob(os.path.join(tensors_dir, "*.npz"))):
        d = os.path.splitext(os.path.basename(f))[0]
        if f"split_{d}" not in keys:
            continue
        t = np.load(f, allow_pickle=False)
        Xa.append(aggregate_features(t["X"], t["mask"]))
        ys.append(t["y"])
        sp.append(z[f"split_{d}"])
        gid.append(z[f"class_{d}"])
    return (np.concatenate(Xa), np.concatenate(ys).astype(np.int64),
            np.concatenate(sp), np.concatenate(gid), classes)


def main() -> None:
    ap = argparse.ArgumentParser(description="Обучение agg-baseline (φ_agg + XGBoost)")
    ap.add_argument("--tensors", default="../tensors/2018")
    ap.add_argument("--splits", default="../splits/splits.npz")
    ap.add_argument("--out", default="../models")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import xgboost as xgb

    print("Сборка агрегатов φ_agg из последовательностей...")
    Xa, y, sp, gid, classes = build(args.tensors, args.splits)
    tr, va, te = sp == 0, sp == 1, sp == 2
    print(f"  train {tr.sum():,} | val {va.sum():,} | test {te.sum():,} | признаков {Xa.shape[1]}")

    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
        early_stopping_rounds=30, n_jobs=-1, random_state=args.seed, tree_method="hist",
    )
    print("Обучение XGBoost (ранняя остановка по val)...")
    clf.fit(Xa[tr], y[tr], eval_set=[(Xa[va], y[va])], verbose=False)
    print(f"  лучших деревьев: {clf.best_iteration + 1}")

    s_va = clf.predict_proba(Xa[va])[:, 1]
    s_te = clf.predict_proba(Xa[te])[:, 1]
    yv, yt = y[va], y[te]

    # рабочие точки по Нейману–Пирсону (на val) — для совместимости с seq-метой
    points = {
        "FPR<=1%": threshold_at_fpr(s_va[yv == 0], 0.01),
        "FPR<=0.1%": threshold_at_fpr(s_va[yv == 0], 0.001),
        "TPR=95% (как этап1)": threshold_at_tpr(s_va[yv == 1], 0.95),
    }
    tau_f1 = best_f1_threshold(s_va, yv)   # операционный порог этапа 1 (argmax-F1 на val)

    # блок метрик этапа 1 на test-2018 (within), порог = argmax-F1 на val
    within = metric_block(s_te, yt, tau_f1)
    mcc_opt, _ = best_mcc(s_te, yt)
    print("\n=== WITHIN-2018 (baseline φ_agg), порог argmax-F1 на val ===")
    print(f"  MCC={within['MCC']:.4f}  F1-macro={within['F1_macro']:.4f}  "
          f"PR-AUC={within['PR_AUC']:.4f}  ROC-AUC={within['ROC_AUC']:.4f}  "
          f"FPR@TPR95={within['FPR_at_TPR95']:.5f}  MCC_опт={mcc_opt:.4f}")
    print("\n  Рабочие точки (порог с val, метрики на test):")
    print(f"  {'точка':<22} {'τ':>7} {'Recall':>8} {'FPR':>9} {'Precision':>10}")
    for name, tau in points.items():
        tpr, fpr, prec = rates(s_te, yt, tau)
        print(f"  {name:<22} {tau:>7.4f} {tpr:>8.4f} {fpr:>9.5f} {prec:>10.4f}")

    os.makedirs(args.out, exist_ok=True)
    clf.save_model(os.path.join(args.out, "baseline_xgb.json"))
    np.savez(os.path.join(args.out, "baseline_meta.npz"),
             model=np.array("xgb_agg"), seq_len=np.array(X_seq_len(args.tensors)),
             feature_names=np.array(FEATURE_NAMES, dtype="U20"),
             classes=np.array(classes, dtype="U40"),
             tau_argmaxf1=np.array(tau_f1),
             tau_fpr01=np.array(points["FPR<=1%"]),
             tau_fpr001=np.array(points["FPR<=0.1%"]),
             tau_tpr95=np.array(points["TPR=95% (как этап1)"]))
    print(f"\n-> сохранено: {args.out}/baseline_xgb.json (+ baseline_meta.npz). "
          f"Эта модель и пороги пойдут на 2017 (infer_2017_baseline.py).")


def X_seq_len(tensors_dir: str) -> int:
    """Длина последовательности тензоров (N) — для записи в мету."""
    f = sorted(glob.glob(os.path.join(tensors_dir, "*.npz")))[0]
    return int(np.load(f, allow_pickle=False)["X"].shape[1])


if __name__ == "__main__":
    main()