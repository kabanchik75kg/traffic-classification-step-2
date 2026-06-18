#!/usr/bin/env python3
"""tensorize_2017.py — тензоризация тестового набора CIC-IDS2017 под модели 2018.

Строит на каждый день ``.npz`` с тензором ``X[поток × N × 3] = [size, dir, IAT]``
ТЕМИ ЖЕ примитивами, что и обучающий 2018 (модуль :mod:`seq_features`), —
это обязательное условие корректности zero-shot переноса модели 2018 → 2017.

Отличия от :mod:`tensorize_2018`.
    * **Прореживания benign нет** (``benign_cap`` отсутствует): 2017 целиком
      выступает тестовой выборкой, берутся все потоки.
    * **Маппинг меток** 2017 → 14 классов 2018: метки приводятся к индексам
      обучающей таксономии. Два класса (``heartbleed``, ``portscan``) в 2018
      отсутствовали — они помечаются кодом −1 (UNSEEN): для бинарной задачи
      это атаки, но мультиклассовая интерпретация к ним неприменима.
    * **Тег дня** — день недели (``Monday``…``Friday``), т.к. 2017 — пять
      суточных дампов, а не посуточные файлы вида ДД-ММ.

Нормализация здесь не применяется (как и в 2018) — см. докстринг
:mod:`seq_features`.

Запуск (venv активен):
    python tensorize_2017.py --data ../data/2017 --out ../tensors/2017

Зависит от: numpy, pyarrow, seq_features.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Делает импорт ядра независимым от текущей рабочей директории.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seq_features import (  # noqa: E402
    FIRST_COL,
    LABEL_COL,
    SRCFILE_COL,
    DIR_COL_CANDIDATES,
    PIAT_COL_CANDIDATES,
    PS_COL_CANDIDATES,
    build_features,
    col_to_matrix,
    is_malicious,
    pick_col,
)

# 14 классов 2018 (порядок = индексы модели). Должен совпадать с обучением.
CLASSES_2018: list[str] = [
    "Benign", "Bot", "DDoS HOIC", "DDoS LOIC-HTTP", "DDoS LOIC-UDP", "DoS GoldenEye",
    "DoS Hulk", "DoS SlowHTTPTest", "DoS Slowloris", "FTP-BruteForce", "SSH-BruteForce",
    "Web BruteForce", "Web SQLi", "Web XSS",
]
IDX2018: dict[str, int] = {c: i for i, c in enumerate(CLASSES_2018)}

# Метка 2017 (нотация LycoS) → имя класса 2018. None = класса нет в 2018 (UNSEEN).
MAP_2017_2018: dict[str, str | None] = {
    "benign": "Benign",
    "bot": "Bot",
    "ddos": "DDoS LOIC-HTTP",
    "dos_goldeneye": "DoS GoldenEye",
    "dos_hulk": "DoS Hulk",
    "dos_slowhttptest": "DoS SlowHTTPTest",
    "dos_slowloris": "DoS Slowloris",
    "ftp_patator": "FTP-BruteForce",
    "ssh_patator": "SSH-BruteForce",
    "webattack_bruteforce": "Web BruteForce",
    "webattack_sql_injection": "Web SQLi",
    "webattack_xss": "Web XSS",
    "heartbleed": None,
    "portscan": None,
}


def day_from_name(path: str) -> str:
    """Имя файла → день недели (``Monday``…``Friday``).

    Для 2017 файлы называются вида ``Wednesday-workingHours.parquet``,
    поэтому берётся часть до первого дефиса.
    """
    return os.path.splitext(os.path.basename(path))[0].split("-")[0]


def labels_to_codes2018(labels: np.ndarray) -> np.ndarray:
    """Метки 2017 → коды классов 2018 (``int16``); UNSEEN → −1.

    Parameters
    ----------
    labels : np.ndarray of str
        Сырые метки 2017 (например, ``"dos_hulk"``, ``"portscan"``).

    Returns
    -------
    np.ndarray
        Коды классов 2018 (индекс в :data:`CLASSES_2018`) либо −1 для классов,
        отсутствовавших в обучении (``heartbleed``, ``portscan``).
    """
    codes = np.full(len(labels), -1, dtype=np.int16)
    for i, lb in enumerate(labels):
        name = MAP_2017_2018.get(str(lb).strip().lower(), None)
        if name is not None:
            codes[i] = IDX2018[name]
    return codes


def process_day(path: str, out_dir: str, N: int) -> int:
    """Тензоризовать один день 2017 (все потоки) и сохранить ``<день недели>.npz``.

    Parameters
    ----------
    path : str
        Путь к размеченному parquet за день.
    out_dir : str
        Папка для выходного ``.npz`` (создаётся при необходимости).
    N : int
        Длина последовательности (должна совпадать с обучением 2018, обычно 60).

    Returns
    -------
    int
        Число обработанных потоков за день.
    """
    day = day_from_name(path)
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    ps_c = pick_col(names, PS_COL_CANDIDATES, "splt_ps")
    dir_c = pick_col(names, DIR_COL_CANDIDATES, "splt_dir")
    piat_c = pick_col(names, PIAT_COL_CANDIDATES, "splt_piat_ms")
    has_file = SRCFILE_COL in names
    has_first = FIRST_COL in names
    cols = [ps_c, dir_c, piat_c, LABEL_COL]
    if has_file:
        cols.append(SRCFILE_COL)
    if has_first:
        cols.append(FIRST_COL)

    Xs, masks, ys, native, codes2018, files, firsts = [], [], [], [], [], [], []
    for batch in pf.iter_batches(batch_size=200_000, columns=cols):
        labels = batch.column(LABEL_COL).to_numpy(zero_copy_only=False).astype(str)
        take = pa.array(np.arange(len(labels)))  # берём ВСЁ (benign_cap отсутствует)

        ps = col_to_matrix(batch.column(ps_c).take(take), N)
        di = col_to_matrix(batch.column(dir_c).take(take), N)
        pi = col_to_matrix(batch.column(piat_c).take(take), N)
        X, mask = build_features(ps, di, pi)

        Xs.append(X)
        masks.append(mask)
        ys.append(is_malicious(labels).astype(np.uint8))
        native.append(labels)
        codes2018.append(labels_to_codes2018(labels))
        if has_file:
            files.append(batch.column(SRCFILE_COL).take(take).to_numpy(zero_copy_only=False).astype(str))
        if has_first:
            firsts.append(batch.column(FIRST_COL).take(take).to_numpy(zero_copy_only=False).astype(np.int64))

    X = np.concatenate(Xs)
    mask = np.concatenate(masks)
    y = np.concatenate(ys)
    native = np.concatenate(native)
    codes2018 = np.concatenate(codes2018)
    files = np.concatenate(files) if files else np.array([""] * len(y))
    firsts = np.concatenate(firsts) if firsts else np.zeros(len(y), dtype=np.int64)
    file_names, file_codes = np.unique(files, return_inverse=True)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{day}.npz")
    np.savez_compressed(
        out_path, X=X, mask=mask, y=y,
        label_codes2018=codes2018, label_classes2018=np.array(CLASSES_2018, dtype="U40"),
        native_label=native.astype("U40"),
        file_codes=file_codes.astype(np.int32), file_names=file_names.astype("U80"),
        first_seen_ms=firsts, day=np.array(day), N=np.array(N),
    )
    n_unseen = int((codes2018 == -1).sum())
    print(f"  {day}: X={X.shape} y(mal={int(y.sum()):,}/{len(y):,}) "
          f"unseen(hb/ps)={n_unseen:,} -> {out_path}")
    return len(y)


def main() -> None:
    """Точка входа: тензоризовать все дни 2017 под N (по умолчанию 60)."""
    ap = argparse.ArgumentParser(description="Тензоризация CIC-IDS2017 под модели 2018")
    ap.add_argument("--data", default="../data/2017")
    ap.add_argument("--out", default="../tensors/2017")
    ap.add_argument("--glob", default="*.parquet")
    ap.add_argument("--N", type=int, default=60)
    args = ap.parse_args()

    data, out = os.path.expanduser(args.data), os.path.expanduser(args.out)
    files = sorted(glob.glob(os.path.join(data, args.glob)))
    if not files:
        raise SystemExit(f"нет parquet в {data}")
    print(f"файлов: {len(files)} | N={args.N} | benign_cap=0 (все benign)")
    tot = sum(process_day(f, out, args.N) for f in files)
    print(f"ВСЕГО: {tot:,} потоков -> {out}")


if __name__ == "__main__":
    main()
