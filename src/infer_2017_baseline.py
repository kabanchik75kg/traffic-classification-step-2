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
import json
import os
import sys
from datetime import datetime

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


def save_results(out_dir: str, within: dict, cross: dict, unseen: dict,
                 args, log_lines: list):
    """Сохранить все метрики и лог в файлы (дописывая к существующим)."""
    os.makedirs(out_dir, exist_ok=True)

    # ---- JSON ----
    json_path = os.path.join(out_dir, "metrics.json")
    # Загружаем существующий JSON, если есть, иначе создаём новый
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = {
            "timestamp": datetime.now().isoformat(),
            "command": " ".join(sys.argv),
            "args": vars(args),
            "within_2018": {},
            "cross_2017": {},
            "unseen_classes_2017": {},
        }
    # Добавляем/обновляем записи для baseline
    data["within_2018"]["φ_agg"] = within
    data["cross_2017"]["φ_agg"] = cross
    data["unseen_classes_2017"]["φ_agg"] = unseen
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=float)

    # ---- CSV (плоская таблица) ----
    csv_path = os.path.join(out_dir, "metrics.csv")
    header = ["model", "split", "MCC", "F1_macro", "PR_AUC", "ROC_AUC", "FPR_at_TPR95", "MCC_opt"]
    # Если CSV существует, читаем заголовки и дописываем строки, иначе создаём
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            lines = f.read().strip().splitlines()
            if lines:
                # предполагаем, что первая строка - заголовок
                existing_models = set()
                for line in lines[1:]:
                    if line:
                        existing_models.add(line.split(",")[0])
                # Если φ_agg уже есть, не дублируем
                if "φ_agg" in existing_models:
                    # можно обновить, но проще перезаписать
                    pass
    # Собираем строки для baseline
    row_within = ["φ_agg", "within", within["MCC"], within["F1_macro"],
                  within["PR_AUC"], within["ROC_AUC"],
                  within["FPR_at_TPR95"], ""]
    row_cross = ["φ_agg", "cross", cross["MCC"], cross["F1_macro"],
                 cross["PR_AUC"], cross["ROC_AUC"],
                 cross["FPR_at_TPR95"], cross["MCC_opt"]]
    # Записываем заново (сохраняя старые строки) или дописываем
    # Проще перезаписать весь CSV, собрав все строки из JSON
    all_rows = []
    for model in data["within_2018"].keys():
        r = [model, "within", data["within_2018"][model]["MCC"],
             data["within_2018"][model]["F1_macro"],
             data["within_2018"][model]["PR_AUC"],
             data["within_2018"][model]["ROC_AUC"],
             data["within_2018"][model]["FPR_at_TPR95"], ""]
        all_rows.append(r)
    for model in data["cross_2017"].keys():
        r = [model, "cross", data["cross_2017"][model]["MCC"],
             data["cross_2017"][model]["F1_macro"],
             data["cross_2017"][model]["PR_AUC"],
             data["cross_2017"][model]["ROC_AUC"],
             data["cross_2017"][model]["FPR_at_TPR95"],
             data["cross_2017"][model]["MCC_opt"]]
        all_rows.append(r)
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in all_rows:
            f.write(",".join(map(str, r)) + "\n")

    # ---- Текстовый лог (дописываем) ----
    log_path = os.path.join(out_dir, "results.log")
    with open(log_path, "a") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"\nРезультаты baseline добавлены в {out_dir}:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    print(f"  - {log_path}")


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

    tau = best_f1_threshold(s_va, yv)              # порог по val-2018
    within = metric_block(s_te, yt, tau)
    cross = metric_block(s17, y17, tau)
    cross_mcc_opt, _ = best_mcc(s17, y17)

    log_lines = []
    log_lines.append(f"\n  φ_agg: τ(argmax-F1, val-2018)={tau:.4f}")
    log_lines.append("\n================ WITHIN-2018 (test) ================")
    log_lines.append(f"  {'модель':<13} {'MCC':>7} {'F1-macro':>9} {'PR-AUC':>8} {'ROC-AUC':>8} {'FPR@TPR95':>12}")
    log_lines.append(f"  {'φ_agg':<13} {within['MCC']:>7.4f} {within['F1_macro']:>9.4f} {within['PR_AUC']:>8.4f} "
                     f"{within['ROC_AUC']:>8.4f} {within['FPR_at_TPR95']:>12.5f}")
    log_lines.append("\n================ CROSS-2017 (перенос) ==============")
    log_lines.append(f"  {'модель':<13} {'MCC_фикс':>8} {'MCC_опт':>8} {'F1-macro':>9} {'PR-AUC':>8} {'ROC-AUC':>8} {'FPR@TPR95':>12}")
    log_lines.append(f"  {'φ_agg':<13} {cross['MCC']:>8.4f} {cross_mcc_opt:>8.4f} {cross['F1_macro']:>9.4f} "
                     f"{cross['PR_AUC']:>8.4f} {cross['ROC_AUC']:>8.4f} {cross['FPR_at_TPR95']:>12.5f}")

    log_lines.append("\n===== строка для сводки 'как Таблица 5.1' (допиши к seq-строкам) =====")
    log_lines.append(f"  {'φ_agg':<13} {within['MCC']:>8.4f} {cross['MCC']:>14.4f} "
                     f"{cross_mcc_opt:>13.4f} {cross['ROC_AUC']:>13.4f}")

    low = np.char.lower(nat17.astype(str))
    unseen = {}
    for cls in ("portscan", "heartbleed"):
        mm = low == cls
        if mm.any():
            val = (s17[mm] >= tau).mean()
            unseen[cls] = float(val)
            log_lines.append(f"  2017 UNSEEN: {cls} = {val:.4f} (n={int(mm.sum()):,})")
    if unseen:
        log_lines.append("\n  2017 UNSEEN (порог argmax-F1 val-2018):  φ_agg  " +
                         "  ".join(f"{k}={v:.4f}" for k, v in unseen.items()))

    out2017 = os.path.join(args.out, "2017")
    os.makedirs(out2017, exist_ok=True)
    np.savez_compressed(os.path.join(out2017, "baseline_ALL.npz"),
                        score=s17, y=y17.astype(np.int8), native_label=nat17, tau=np.array(tau))
    log_lines.append(f"\n  -> {out2017}/baseline_ALL.npz")

    # Выводим в консоль
    for line in log_lines:
        print(line)

    # Сохраняем в файлы
    save_results(
        out_dir=args.out,
        within=within,
        cross={**cross, "MCC_opt": cross_mcc_opt},  # добавим опт. порог
        unseen=unseen,
        args=args,
        log_lines=log_lines
    )


if __name__ == "__main__":
    main()