#!/usr/bin/env python3
"""tensorize_2018.py — тензоризация обучающего набора CSE-CIC-IDS2018.

Из размеченных посуточных parquet (NFStream SPLT, записано до 100 пакетов
на поток) строит на каждый день ``.npz`` с тензором ``X[поток × N × 3] =
[size, dir, IAT]``, маской реальных пакетов, бинарной меткой benign/malicious
и метаданными (день, файл-источник, время, класс) — последние нужны для
сплита без утечки и для разбора метрик по классам.

Прежнее имя файла: ``tensorize_sequences.py`` (переименован для симметрии
с ``tensorize_2017.py``). Числовое поведение НЕ изменено — общие примитивы
вынесены в :mod:`seq_features`, остальное сохранено дословно.

Ключевые решения.
    * **N = 60** записывается один раз. Абляция N ∈ {20, 40, 60} выполняется
      на обучении срезом ``X[:, :20]`` / ``X[:, :40]`` (маска это учитывает),
      перепарсивать parquet не нужно.
    * **benign_cap** прореживает benign-потоки до заданного числа на день
      (по умолчанию 500 000), атаки сохраняются полностью. Это сознательно
      повышает долю атак в обучающем наборе; прореживание детерминировано
      (``seed``), поэтому воспроизводимо. Для тестового 2017 прореживания нет.
    * Нормализация здесь не применяется — см. докстринг :mod:`seq_features`.

Запуск (venv активен):
    python tensorize_2018.py --data ../data/2018 --out ../tensors/2018 --glob 'Wednesday-14-02'
    python tensorize_2018.py --data ../data/2018 --out ../tensors/2018

Зависит от: numpy, pyarrow, seq_features. Память не растёт с размером дня
(parquet читается батчами).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Делает импорт ядра независимым от текущей рабочей директории.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seq_features import (  # noqa: E402
    FIRST_COL,
    LABEL_COL,
    NPKT_COL,
    SRCFILE_COL,
    DIR_COL_CANDIDATES,
    PIAT_COL_CANDIDATES,
    PS_COL_CANDIDATES,
    build_features,
    col_to_matrix,
    is_malicious,
    pick_col,
)


def day_from_name(path: str) -> str:
    """Имя файла parquet → короткий тег дня вида ``"14-02"``.

    Извлекает шаблон ``ДД-ММ`` (например, из ``Wednesday-14-02-2018_labeled``);
    при отсутствии шаблона возвращает имя файла без расширения.
    """
    m = re.search(r"(\d{2}-\d{2})", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]


def pct(a: np.ndarray, q: float) -> int:
    """Перцентиль ``q`` массива ``a`` как целое (0 для пустого массива)."""
    return int(np.percentile(a, q)) if len(a) else 0


def stats_pass(pf: "pq.ParquetFile") -> tuple[int, int, int]:
    """Лёгкий проход по файлу: распределение меток и длин потоков.

    Печатает суммарные счётчики (всего / атак / benign), CDF длины потока
    (p50/p90/p95/p99/max), долю потоков длиннее 100 пакетов (обрезаются SPLT)
    и медиану/p95 длины по каждому классу. Нужен, в частности, для выбора N
    по эмпирической CDF и для оценки доли прореживания benign.

    Parameters
    ----------
    pf : pyarrow.parquet.ParquetFile
        Открытый размеченный parquet за один день.

    Returns
    -------
    tuple[int, int, int]
        ``(всего_потоков, атак, benign)``.
    """
    labels_all, npkt_all = [], []
    for batch in pf.iter_batches(batch_size=500_000, columns=[LABEL_COL, NPKT_COL]):
        labels_all.append(batch.column(LABEL_COL).to_numpy(zero_copy_only=False).astype(str))
        npkt_all.append(batch.column(NPKT_COL).to_numpy(zero_copy_only=False).astype(np.int64))
    labels = np.concatenate(labels_all)
    npkt = np.concatenate(npkt_all)
    mal = is_malicious(labels)
    n_total, n_mal = len(labels), int(mal.sum())
    n_ben = n_total - n_mal
    print(f"  всего потоков : {n_total:,}")
    print(f"  атак          : {n_mal:,}")
    print(f"  benign        : {n_ben:,}")
    print(f"  длина потока (пакетов): p50={pct(npkt,50)} p90={pct(npkt,90)} "
          f"p95={pct(npkt,95)} p99={pct(npkt,99)} max={int(npkt.max())}")
    print(f"  доля потоков >100 пакетов (обрезаны SPLT): {100*np.mean(npkt>100):.2f}%")
    print("  длина по классам (медиана / p95 / n):")
    for cls in sorted(set(labels)):
        m = labels == cls
        print(f"     {cls:<22} медиана={pct(npkt[m],50):<5} p95={pct(npkt[m],95):<6} n={int(m.sum()):,}")
    return n_total, n_mal, n_ben


def process_day(
    path: str, out_dir: str, N: int, benign_cap: int, seed: int
) -> dict:
    """Тензоризовать один день и сохранить ``<день>.npz``.

    Шаги: статистический проход (:func:`stats_pass`) → вычисление доли
    сохраняемых benign → батчевый проход с прореживанием benign и сборкой
    тензора (:func:`seq_features.build_features`) → запись ``.npz`` с тензором,
    маской, метками и метаданными.

    Parameters
    ----------
    path : str
        Путь к размеченному parquet за день.
    out_dir : str
        Папка для выходного ``.npz`` (создаётся при необходимости).
    N : int
        Длина последовательности (по максимуму абляции, обычно 60).
    benign_cap : int
        Максимум benign-потоков на день; ``<= 0`` — сохранять все.
    seed : int
        Зерно ГПСЧ для воспроизводимого прореживания benign.

    Returns
    -------
    dict
        Сводка дня: ``{"day", "n", "n_mal", "shape"}``.
    """
    day = day_from_name(path)
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    ps_c = pick_col(names, PS_COL_CANDIDATES, "splt_ps")
    dir_c = pick_col(names, DIR_COL_CANDIDATES, "splt_dir")
    piat_c = pick_col(names, PIAT_COL_CANDIDATES, "splt_piat_ms")
    has_file = SRCFILE_COL in names
    has_first = FIRST_COL in names

    print(f"\n=== {day} :: {os.path.basename(path)} ===")
    n_total, n_mal, n_ben = stats_pass(pf)
    keep_p = 1.0 if (benign_cap <= 0 or n_ben <= benign_cap) else benign_cap / n_ben
    print(f"  benign оставляем p={keep_p:.4f} (~{int(n_ben*keep_p):,})")

    rng = np.random.default_rng(seed)
    cols = [ps_c, dir_c, piat_c, LABEL_COL]
    if has_file:
        cols.append(SRCFILE_COL)
    if has_first:
        cols.append(FIRST_COL)

    Xs, masks, ys, labs, files, firsts = [], [], [], [], [], []
    for batch in pf.iter_batches(batch_size=200_000, columns=cols):
        labels = batch.column(LABEL_COL).to_numpy(zero_copy_only=False).astype(str)
        mal = is_malicious(labels)
        keep = mal | (rng.random(len(labels)) < keep_p)
        if not keep.any():
            continue
        idx = np.where(keep)[0]
        take = pa.array(idx)  # отбираем ДО разбора splt-массивов — это и есть ускорение

        ps = col_to_matrix(batch.column(ps_c).take(take), N)
        di = col_to_matrix(batch.column(dir_c).take(take), N)
        pi = col_to_matrix(batch.column(piat_c).take(take), N)
        X, mask = build_features(ps, di, pi)

        Xs.append(X)
        masks.append(mask)
        ys.append(mal[idx].astype(np.uint8))
        labs.append(labels[idx])
        if has_file:
            files.append(batch.column(SRCFILE_COL).take(take).to_numpy(zero_copy_only=False).astype(str))
        if has_first:
            firsts.append(batch.column(FIRST_COL).take(take).to_numpy(zero_copy_only=False).astype(np.int64))

    X = np.concatenate(Xs)
    mask = np.concatenate(masks)
    y = np.concatenate(ys)
    labels = np.concatenate(labs)
    files = np.concatenate(files) if files else np.array([""] * len(y))
    firsts = np.concatenate(firsts) if firsts else np.zeros(len(y), dtype=np.int64)

    label_classes, label_codes = np.unique(labels, return_inverse=True)
    file_names, file_codes = np.unique(files, return_inverse=True)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{day}.npz")
    np.savez_compressed(
        out_path, X=X, mask=mask, y=y,
        label_codes=label_codes.astype(np.int16), label_classes=label_classes.astype("U40"),
        file_codes=file_codes.astype(np.int32), file_names=file_names.astype("U80"),
        first_seen_ms=firsts, day=np.array(day), N=np.array(N),
        benign_cap=np.array(benign_cap), seed=np.array(seed),
    )
    print(f"  -> {out_path}  X={X.shape} {X.dtype}  y(mal={int(y.sum()):,}/{len(y):,})")
    return dict(day=day, n=len(y), n_mal=int(y.sum()), shape=X.shape)


def main() -> None:
    """Точка входа: тензоризовать все parquet по маске и напечатать итог."""
    ap = argparse.ArgumentParser(description="Тензоризация sequence-представления (2018)")
    ap.add_argument("--data", default="../data/2018", help="папка с *.parquet")
    ap.add_argument("--out", default="../tensors/2018", help="папка для *.npz")
    ap.add_argument("--glob", default="*.parquet", help="маска файлов")
    ap.add_argument("--N", type=int, default=60, help="длина (по максимуму абляции)")
    ap.add_argument("--benign-cap", type=int, default=500_000, help="макс. benign/день (0=все)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pattern = args.glob if args.glob.endswith(".parquet") else args.glob + "*.parquet"
    files = sorted(glob.glob(os.path.join(args.data, pattern)))
    if not files:
        raise SystemExit(f"Не найдено файлов по маске {pattern} в {args.data}")
    print(f"Файлов: {len(files)} | N={args.N} | benign-cap={args.benign_cap}")

    summary = [process_day(f, args.out, args.N, args.benign_cap, args.seed) for f in files]

    print("\n================ ИТОГ ================")
    tot = tot_mal = 0
    for s in summary:
        tot += s["n"]
        tot_mal += s["n_mal"]
        print(f"  {s['day']}: {s['n']:,} потоков (атак {s['n_mal']:,})  X={s['shape']}")
    print(f"  ВСЕГО: {tot:,} потоков, атак {tot_mal:,} ({100*tot_mal/max(tot,1):.1f}%)")


if __name__ == "__main__":
    main()
