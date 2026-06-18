#!/usr/bin/env python3
"""infer_2017_baseline.py — перенос agg-baseline (φ_agg + XGBoost) на 2017.

Симметрично infer/evaluate для seq-моделей: грузит сохранённую модель
``baseline_xgb.json``, считает φ_agg на тех же тензорах (2018 val/test и весь
2017), и печатает блок метрик этапа 1 (within-2018 и cross-2017) в ОДНОМ
формате со строками seq — чтобы строка «φ_agg» встала рядом с ними в сводке.

Протокол идентичен seq: порог = argmax-F1 на val-2018, фиксируется и
применяется к test-2018 и ко всему 2017 (слепой перенос). φ_agg не нормируется
(деревьям нормализация не нужна), поэтому μ/σ здесь не участвуют — представление
агрегатное, и в этом весь смысл сравнения.

Запуск:
    python infer_2017_baseline.py --tensors2018 ../tensors/2018 \
        --tensors2017 ../tensors/2017 --splits ../splits/splits.npz \
        --models-dir ../models --out ../results

Зависит от: numpy, xgboost, metrics_np, train_baseline.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from metrics_np import best_f1_threshold, best_mcc, metric_block
from train_baseline import aggregate_features, build


def build_2017(tensors_dir: str):
    """Собрать φ_agg всего 2017 + метку и нативные метки (для UNSEEN-строки)."""
    Xa, ys, nats = [], [], []
    for f in sorted(glob.glob(os.path.join(tensors_dir, "*.npz"))):
        t = np.load(f, allow_pickle=False)
        Xa.append(aggregate_features(t["X"], t["mask"]))
        ys.append(t["y"].astype(np.int64))
        nat = t["native_label"] if "native_label" in t.files else np.array([""] * len(t["y"]))
        nats.append(nat.astype("U40"))
    return np.concatenate(Xa), np.concatenate(ys), np.concatenate(nats)


def main() -> None:
    ap = argparse.ArgumentParser(description="Перенос agg-baseline на 2017")
    ap.add_argument("--tensors2018", default="../tensors/2018")
    ap.add_argument("--tensors2017", default="../tensors/2017")
    ap.add_argument("--splits", default="../splits/splits.npz")
    ap.add_argument("--models-dir", default="../models")
    ap.add_argument("--out", default="../results")
    args = ap.parse_args()

    import xgboost as xgb

    clf = xgb.XGBClassifier()
    clf.load_model(os.path.join(args.models_dir, "baseline_xgb.json"))

    print("Сборка φ_agg: 2018 (val/test) ...")
    Xa, y, sp, gid, classes = build(args.tensors2018, args.splits)
    va, te = sp == 1, sp == 2
    s_va = clf.predict_proba(Xa[va])[:, 1]
    s_te = clf.predict_proba(Xa[te])[:, 1]
    yv, yt = y[va], y[te]

    print("Сборка φ_agg: весь 2017 ...")
    Xa17, y17, nat17 = build_2017(args.tensors2017)
    s17 = clf.predict_proba(Xa17)[:, 1]
    print(f"  2017: {len(y17):,} потоков (атак {int(y17.sum()):,}, {100*y17.mean():.1f}%)")

    tau = best_f1_threshold(s_va, yv)              # порог по val-2018 (как seq)
    within = metric_block(s_te, yt, tau)
    cross = metric_block(s17, y17, tau)
    cross_mcc_opt, _ = best_mcc(s17, y17)

    print(f"\n  φ_agg: τ(argmax-F1, val-2018)={tau:.4f}")
    print("\n================ WITHIN-2018 (test) ================")
    print(f"  {'модель':<13} {'MCC':>7} {'F1-macro':>9} {'PR-AUC':>8} {'ROC-AUC':>8} {'FPR@TPR95':>12}")
    print(f"  {'φ_agg':<13} {within['MCC']:>7.4f} {within['F1_macro']:>9.4f} {within['PR_AUC']:>8.4f} "
          f"{within['ROC_AUC']:>8.4f} {within['FPR_at_TPR95']:>12.5f}")
    print("\n================ CROSS-2017 (перенос) ==============")
    print(f"  {'модель':<13} {'MCC_фикс':>8} {'MCC_опт':>8} {'F1-macro':>9} {'PR-AUC':>8} {'ROC-AUC':>8} {'FPR@TPR95':>12}")
    print(f"  {'φ_agg':<13} {cross['MCC']:>8.4f} {cross_mcc_opt:>8.4f} {cross['F1_macro']:>9.4f} "
          f"{cross['PR_AUC']:>8.4f} {cross['ROC_AUC']:>8.4f} {cross['FPR_at_TPR95']:>12.5f}")

    print("\n===== строка для сводки 'как Таблица 5.1' (допиши к seq-строкам) =====")
    print(f"  {'φ_agg':<13} {within['MCC']:>8.4f} {cross['MCC']:>14.4f} "
          f"{cross_mcc_opt:>13.4f} {cross['ROC_AUC']:>13.4f}")

    low = np.char.lower(nat17.astype(str))
    unseen = []
    for cls in ("portscan", "heartbleed"):
        mm = low == cls
        if mm.any():
            unseen.append(f"{cls}={ (s17[mm] >= tau).mean():.4f} (n={int(mm.sum()):,})")
    if unseen:
        print("\n  2017 UNSEEN (порог argmax-F1 val-2018):  φ_agg  " + "  ".join(unseen))

    out2017 = os.path.join(args.out, "2017")
    os.makedirs(out2017, exist_ok=True)
    np.savez_compressed(os.path.join(out2017, "baseline_ALL.npz"),
                        score=s17, y=y17.astype(np.int8), native_label=nat17, tau=np.array(tau))
    print(f"\n  -> {out2017}/baseline_ALL.npz")


if __name__ == "__main__":
    main()