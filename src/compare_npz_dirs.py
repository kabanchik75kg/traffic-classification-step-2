#!/usr/bin/env python3
"""compare_npz_dirs.py — поэлементное сравнение двух папок с ``.npz``.

Назначение — подтвердить, что рефакторинг тензоризаторов (вынос ядра в
:mod:`seq_features`) НЕ изменил результат: новые ``.npz`` должны совпадать
со старыми по каждому массиву.

Почему нельзя сравнивать файлы побайтово.
    ``.npz`` — это zip-контейнер, и в его заголовки записываются временны́е
    метки создания. Поэтому два архива с идентичным содержимым всё равно
    различаются как файлы. Корректное сравнение — поэлементное: открыть оба
    архива и сверить каждый массив через ``np.array_equal``.

Запуск:
    python compare_npz_dirs.py ../tensors/2018 ../tensors/2018_new
    python compare_npz_dirs.py ../tensors/2017 ../tensors/2017_new

Код возврата: 0 — всё совпало, 1 — есть расхождения (удобно для CI).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np


def compare_one(old_path: str, new_path: str) -> list[str]:
    """Сравнить два ``.npz`` по всем ключам и вернуть список расхождений.

    Parameters
    ----------
    old_path, new_path : str
        Пути к старому и новому архивам одного дня.

    Returns
    -------
    list[str]
        Человекочитаемые описания расхождений; пустой список — полное совпадение.
    """
    a = np.load(old_path, allow_pickle=False)
    b = np.load(new_path, allow_pickle=False)
    issues: list[str] = []

    keys_a, keys_b = set(a.files), set(b.files)
    if keys_a != keys_b:
        only_a = sorted(keys_a - keys_b)
        only_b = sorted(keys_b - keys_a)
        if only_a:
            issues.append(f"ключи только в старом: {only_a}")
        if only_b:
            issues.append(f"ключи только в новом: {only_b}")

    for k in sorted(keys_a & keys_b):
        x, y = a[k], b[k]
        if x.shape != y.shape:
            issues.append(f"{k}: форма {x.shape} != {y.shape}")
            continue
        if x.dtype != y.dtype:
            issues.append(f"{k}: тип {x.dtype} != {y.dtype}")
        if not np.array_equal(x, y):
            n_diff = int((x != y).sum()) if x.shape == y.shape else -1
            issues.append(f"{k}: значения различаются (ячеек: {n_diff:,})")
    return issues


def main() -> None:
    """Сверить все одноимённые ``.npz`` в двух папках; печать итога, код возврата."""
    if len(sys.argv) != 3:
        raise SystemExit("использование: python compare_npz_dirs.py <старая_папка> <новая_папка>")
    old_dir, new_dir = sys.argv[1], sys.argv[2]

    old_files = {os.path.basename(p): p for p in glob.glob(os.path.join(old_dir, "*.npz"))}
    new_files = {os.path.basename(p): p for p in glob.glob(os.path.join(new_dir, "*.npz"))}
    common = sorted(set(old_files) & set(new_files))
    if not common:
        raise SystemExit(f"нет одноимённых .npz в {old_dir} и {new_dir}")

    missing = sorted(set(old_files) ^ set(new_files))
    if missing:
        print(f"[!] есть несовпадающие по наличию файлы: {missing}")

    total_issues = 0
    for name in common:
        issues = compare_one(old_files[name], new_files[name])
        if issues:
            total_issues += len(issues)
            print(f"✗ {name}")
            for it in issues:
                print(f"    - {it}")
        else:
            print(f"✓ {name} — идентичен")

    print("-" * 50)
    if total_issues == 0 and not missing:
        print("ИТОГ: все тензоры идентичны — рефакторинг безопасен.")
        sys.exit(0)
    else:
        print(f"ИТОГ: расхождений {total_issues}, несовпадений по файлам {len(missing)}.")
        sys.exit(1)


if __name__ == "__main__":
    main()