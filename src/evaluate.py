#!/usr/bin/env python3
"""evaluate.py — единый оценщик: within-2018 и cross-2017 в метриках этапа 1.

Загружает обученные sequence-модели (``../models/seq_<model>.pt`` + мета),
перескоривает валидацию и тест 2018 и весь 2017, и для каждой модели печатает
блок метрик Таблицы 4.1 этапа 1:

    MCC, F1-macro, PR-AUC, ROC-AUC, FPR при TPR=0,95.

Протокол (как в этапе 1):
    * порог фиксируется по argmax-F1 на ВАЛИДАЦИИ 2018 и применяется без
      подстройки к test-2018 (within) и ко всему 2017 (cross — слепой перенос);
    * нормализация 2017 — теми же μ/σ из ``splits.npz`` (train-2018), 1:1 как
      на обучении, тензоры 2017 не модифицируются;
    * пороговонезависимые метрики (PR-AUC, ROC-AUC, FPR@TPR=0,95) корректны
      независимо от порога и доли атак.

Задача бинарная (benign/attack), поэтому F1-macro считается по двум классам —
это бинарный аналог мультиклассового F1-macro этапа 1. portscan/heartbleed в
2017 трактуются как атаки (несигнатурное обобщение) и выводятся отдельной строкой.

Запуск:
    python evaluate.py --models cnn tcn bilstm gru transformer \
        --tensors2018 ../tensors/2018 --tensors2017 ../tensors/2017 \
        --splits ../splits/splits.npz --out ../results

Зависит от: numpy, scikit-learn, torch, seq_models, train_seq.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

import seq_models
from train_seq import load_and_prepare   # загрузчик 2018 (numpy, без torch)
from metrics_np import best_f1_threshold, best_mcc, metric_block   # метрики этапа 1


# ---------------- скоринг и загрузка ----------------
@torch.no_grad()
def score_model(model: torch.nn.Module, X: np.ndarray, M: np.ndarray,
                dev: torch.device, batch: int) -> np.ndarray:
    """Прогнать модель по (X, M) батчами и вернуть вероятности атаки [N]."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch].astype(np.float32)).to(dev)
        mb = torch.from_numpy(M[i:i + batch].astype(np.float32)).to(dev)
        out.append(torch.sigmoid(model(xb, mb)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def load_2017(tensors_dir: str, splits_path: str, seq_len: int):
    """Собрать весь 2017, нормализовать size/IAT теми же μ/σ из splits.npz.

    Returns
    -------
    (X, M, y, native) : тензор float16 [N,seq_len,3], маска uint8, метка y,
    нативные метки 2017 (для строки portscan/heartbleed).
    """
    z = np.load(splits_path, allow_pickle=False)
    mu_s = z["mu_size"].item()      # скаляр
    sg_s = z["sigma_size"].item()
    mu_i = z["mu_iat"].item()
    sg_i = z["sigma_iat"].item()
    Xs, Ms, ys, nats = [], [], [], []
    for f in sorted(glob.glob(os.path.join(tensors_dir, "*.npz"))):
        t = np.load(f, allow_pickle=False)
        X = t["X"][:, :seq_len, :].astype(np.float32)
        mask = t["mask"][:, :seq_len].astype(np.float32)
        X[..., 0] = (X[..., 0] - mu_s) / sg_s * mask     # ровно как train_seq
        X[..., 2] = (X[..., 2] - mu_i) / sg_i * mask
        Xs.append(X.astype(np.float16))
        Ms.append(mask.astype(np.uint8))
        ys.append(t["y"].astype(np.int8))
        if "native_label" in t.files:
            nat = t["native_label"].astype(str)
        else:
            nat = np.array([""] * len(t["y"]), dtype=str)
        nats.append(nat)
    return (np.concatenate(Xs), np.concatenate(Ms),
            np.concatenate(ys), np.concatenate(nats))


def fmt_row(name: str, b: dict) -> str:
    """Отформатировать строку таблицы метрик."""
    return (f"  {name:<13} {b['MCC']:>7.4f} {b['F1_macro']:>9.4f} {b['PR_AUC']:>8.4f} "
            f"{b['ROC_AUC']:>8.4f} {b['FPR_at_TPR95']:>12.5f}")


def save_results(out_dir: str, within: dict, cross: dict, unseen: dict,
                 args, model_names: list, log_lines: list):
    """Сохранить все метрики и лог в файлы."""
    os.makedirs(out_dir, exist_ok=True)

    # ---- JSON ----
    data = {
        "timestamp": datetime.now().isoformat(),
        "command": " ".join(sys.argv),
        "args": vars(args),
        "within_2018": within,
        "cross_2017": cross,
        "unseen_classes_2017": unseen,
    }
    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=float)

    # ---- CSV (плоская таблица) ----
    csv_path = os.path.join(out_dir, "metrics.csv")
    header = ["model", "split", "MCC", "F1_macro", "PR_AUC", "ROC_AUC", "FPR_at_TPR95", "MCC_opt"]
    rows = []
    for model in model_names:
        if model in within:
            r = [model, "within", within[model]["MCC"], within[model]["F1_macro"],
                 within[model]["PR_AUC"], within[model]["ROC_AUC"],
                 within[model]["FPR_at_TPR95"], ""]
            rows.append(r)
        if model in cross:
            r = [model, "cross", cross[model]["MCC"], cross[model]["F1_macro"],
                 cross[model]["PR_AUC"], cross[model]["ROC_AUC"],
                 cross[model]["FPR_at_TPR95"], cross[model]["MCC_opt"]]
            rows.append(r)
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    # ---- Текстовый лог (дублирует консоль) ----
    log_path = os.path.join(out_dir, "results.log")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"\nРезультаты сохранены в {out_dir}:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    print(f"  - {log_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="within-2018 и cross-2017 в метриках этапа 1")
    ap.add_argument("--models", nargs="+", default=seq_models.MODEL_NAMES)
    ap.add_argument("--models-dir", default="../models")
    ap.add_argument("--tensors2018", default="../tensors/2018")
    ap.add_argument("--tensors2017", default="../tensors/2017")
    ap.add_argument("--splits", default="../splits/splits.npz")
    ap.add_argument("--out", default="../results")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--seq-len", type=int, default=60)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_lines = [f"Устройство: {dev}"]
    print(log_lines[-1])

    print("Загрузка 2018 (val/test) + нормализация на лету...")
    X16, M, y, split, gid, classes = load_and_prepare(args.tensors2018, args.splits, args.seq_len)
    iv, it = np.where(split == 1)[0], np.where(split == 2)[0]
    yv, yt = y[iv].astype(int), y[it].astype(int)
    log_lines.append(f"  val {len(iv):,} (атак {yv.sum():,}) | test {len(it):,} (атак {yt.sum():,})")
    print(log_lines[-1])

    print("Загрузка всего 2017 + нормализация теми же μ/σ...")
    X17, M17, y17, nat17 = load_2017(args.tensors2017, args.splits, args.seq_len)
    y17 = y17.astype(int)
    log_lines.append(f"  2017: {len(y17):,} потоков (атак {y17.sum():,}, {100*y17.mean():.1f}%)")
    print(log_lines[-1])

    out2017 = os.path.join(args.out, "2017")
    os.makedirs(out2017, exist_ok=True)
    within_rows, cross_rows = {}, {}
    unseen_rows = {}   # для unseen классов

    for m in args.models:
        pt = os.path.join(args.models_dir, f"seq_{m}.pt")
        meta_p = os.path.join(args.models_dir, f"seq_{m}_meta.npz")
        if not (os.path.exists(pt) and os.path.exists(meta_p)):
            log_lines.append(f"  [!] пропуск {m}: нет {pt} или меты")
            print(log_lines[-1])
            continue
        meta = np.load(meta_p, allow_pickle=True)
        sl = int(meta["seq_len"])
        model = seq_models.make_model(m, sl).to(dev)
        model.load_state_dict(torch.load(pt, map_location=dev))

        sv = score_model(model, X16[iv][:, :sl], M[iv][:, :sl], dev, args.batch)
        st = score_model(model, X16[it][:, :sl], M[it][:, :sl], dev, args.batch)
        s17 = score_model(model, X17[:, :sl], M17[:, :sl], dev, args.batch)

        tau = best_f1_threshold(sv, yv)       # порог по val-2018 (как этап 1)
        within_rows[m] = metric_block(st, yt, tau)
        cr = metric_block(s17, y17, tau)      # перенос при фиксированном пороге
        cr["MCC_opt"], _ = best_mcc(s17, y17)  # перенос при оптимальном пороге (оракул)
        cross_rows[m] = cr

        np.savez_compressed(os.path.join(out2017, f"{m}_ALL.npz"),
                            score=s17, y=y17.astype(np.int8), native_label=nat17, tau=np.array(tau))
        log_lines.append(f"  {m:<12} τ(argmax-F1, val-2018)={tau:.4f}")
        print(log_lines[-1])

        # ---- сохраняем also scores для 2018 (по желанию) ----
        # np.savez_compressed(os.path.join(args.out, f"{m}_2018_scores.npz"),
        #                     val_score=sv, val_y=yv, test_score=st, test_y=yt, tau=tau)

    # ----- сбор метрик unseen (portscan/heartbleed) -----
    low = np.char.lower(nat17.astype(str))
    for m in args.models:
        f = os.path.join(out2017, f"{m}_ALL.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=False)
        s, tau = z["score"], float(z["tau"])
        line = [f"  {m:<12}"]
        for cls in ("portscan", "heartbleed"):
            mm = low == cls
            if mm.any():
                line.append(f"{cls}={ (s[mm] >= tau).mean():.4f} (n={int(mm.sum()):,})")
        if len(line) > 1:
            line_str = " ".join(line)
            log_lines.append(line_str)
            print(line_str)
            unseen_rows[m] = {cls: float((s[low == cls] >= tau).mean()) for cls in ("portscan", "heartbleed") if (low == cls).any()}

    head_within = f"  {'модель':<13} {'MCC':>7} {'F1-macro':>9} {'PR-AUC':>8} {'ROC-AUC':>8} {'FPR@TPR95':>12}"
    log_lines.append("\n================ WITHIN-2018 (test) ================")
    log_lines.append(head_within)
    for m in args.models:
        if m in within_rows:
            log_lines.append(fmt_row(m, within_rows[m]))
    log_lines.append("\n================ CROSS-2017 (перенос) ==============")
    log_lines.append(f"  {'модель':<13} {'MCC_фикс':>8} {'MCC_опт':>8} {'F1-macro':>9} {'PR-AUC':>8} {'ROC-AUC':>8} {'FPR@TPR95':>12}")
    for m in args.models:
        if m in cross_rows:
            b = cross_rows[m]
            log_lines.append(f"  {m:<13} {b['MCC']:>8.4f} {b['MCC_opt']:>8.4f} {b['F1_macro']:>9.4f} "
                             f"{b['PR_AUC']:>8.4f} {b['ROC_AUC']:>8.4f} {b['FPR_at_TPR95']:>12.5f}")

    # компактная сводка
    log_lines.append("\n===== СВОДКА как Таблица 5.1 этапа 1 (MCC) =====")
    log_lines.append(f"  {'модель':<13} {'within':>8} {'перенос(фикс)':>14} {'перенос(опт)':>13} {'ROC-AUC(пер)':>13}")
    for m in args.models:
        if m in within_rows and m in cross_rows:
            log_lines.append(f"  {m:<13} {within_rows[m]['MCC']:>8.4f} {cross_rows[m]['MCC']:>14.4f} "
                             f"{cross_rows[m]['MCC_opt']:>13.4f} {cross_rows[m]['ROC_AUC']:>13.4f}")

    log_lines.append(f"\nПримечание: within-2018 — при доле атак выборки (≈33%); ROC-AUC и FPR@TPR95 "
                     f"от доли не зависят и прямо сопоставимы с этапом 1. Скоры 2017 -> {out2017}/")

    # теперь выводим всё в консоль (уже напечатано по ходу, но продублируем для полноты)
    for line in log_lines:
        print(line)

    # ----- сохраняем результаты в файлы -----
    save_results(
        out_dir=args.out,
        within=within_rows,
        cross=cross_rows,
        unseen=unseen_rows,
        args=args,
        model_names=args.models,
        log_lines=log_lines
    )


if __name__ == "__main__":
    main()