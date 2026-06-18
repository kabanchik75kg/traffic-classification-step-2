#!/usr/bin/env python3
"""inspect_splits.py — осмотр бандла сплита ``splits.npz`` из терминала.

``prepare_split.py`` сохраняет результат компактно в один архив:
  * ``split_<день>`` — int8 на каждый поток (0=train, 1=val, 2=test);
  * ``class_<день>`` — int16 глобальный id класса на каждый поток;
  * ``classes``      — список имён классов (глобальный словарь);
  * ``mu_size, sigma_size, mu_iat, sigma_iat`` — параметры нормализации (по train);
  * ``seed``.

Скрипт не пересчитывает сплит, а лишь читает готовый файл и печатает:
сырой перечень ключей (форма/тип), размеры train/val/test, покрытие классов
по трём частям и зафиксированные μ/σ нормализации. Удобно, чтобы свериться
с отчётом, не запуская сборку заново.

Запуск:
    python inspect_splits.py --splits ../splits/splits.npz
"""
from __future__ import annotations

import argparse

import numpy as np

SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}


def dump_keys(z: "np.lib.npyio.NpzFile") -> None:
    """Напечатать сырой перечень массивов архива: имя, форму и тип."""
    print("КЛЮЧИ АРХИВА:")
    for k in z.files:
        a = z[k]
        scalar = "" if a.ndim else f" = {a.item()!r}"
        print(f"  {k:<16} shape={str(a.shape):<14} dtype={a.dtype}{scalar}")
    print()


def collect_days(z: "np.lib.npyio.NpzFile") -> list[str]:
    """Список тегов дней по ключам вида ``split_<день>`` (в порядке сортировки)."""
    return sorted(k[len("split_"):] for k in z.files if k.startswith("split_"))


def report_sizes(z: "np.lib.npyio.NpzFile", days: list[str]) -> None:
    """Печать размеров train/val/test (суммарно по всем дням) и долей."""
    totals = {0: 0, 1: 0, 2: 0}
    for d in days:
        split = z[f"split_{d}"]
        for s in (0, 1, 2):
            totals[s] += int((split == s).sum())
    grand = sum(totals.values())
    print("РАЗМЕРЫ СПЛИТОВ:")
    for s in (0, 1, 2):
        share = 100 * totals[s] / grand if grand else 0.0
        print(f"  {SPLIT_NAMES[s]:<5} {totals[s]:>12,}  ({share:.1f}%)")
    print(f"  всего {grand:,}\n")


def report_coverage(z: "np.lib.npyio.NpzFile", days: list[str]) -> None:
    """Печать покрытия каждого класса по частям train/val/test.

    Помечает классы, выпавшие из какой-либо части (потенциальная проблема
    для микроклассов).
    """
    classes = [str(c) for c in z["classes"]]
    counts = {s: {c: 0 for c in classes} for s in (0, 1, 2)}
    for d in days:
        split = z[f"split_{d}"]
        gid = z[f"class_{d}"]
        for s in (0, 1, 2):
            sel = gid[split == s]
            if sel.size:
                vals, cnts = np.unique(sel, return_counts=True)
                for v, c in zip(vals, cnts):
                    counts[s][classes[int(v)]] += int(c)
    print("ПОКРЫТИЕ КЛАССОВ (train / val / test):")
    all_ok = True
    for c in classes:
        row = [counts[s][c] for s in (0, 1, 2)]
        ok = all(x > 0 for x in row)
        all_ok &= ok
        flag = "" if ok else "  <-- ПУСТО В КАКОЙ-ТО ЧАСТИ!"
        print(f"  {c:<20} {row[0]:>11,} / {row[1]:>9,} / {row[2]:>9,}{flag}")
    print("\nВСЕ КЛАССЫ ПОКРЫТЫ ✅" if all_ok else "\nЕСТЬ ПУСТЫЕ КЛАССЫ ❌")
    print()


def report_norm(z: "np.lib.npyio.NpzFile") -> None:
    """Печать зафиксированных параметров нормализации (по train)."""
    print("НОРМАЛИЗАЦИЯ (по train, реальные пакеты):")
    print(f"  size: mu={float(z['mu_size']):.3f}  sigma={float(z['sigma_size']):.3f}")
    print(f"  iat : mu={float(z['mu_iat']):.3f}  sigma={float(z['sigma_iat']):.3f}")
    if "seed" in z.files:
        print(f"  seed={int(z['seed'])}")
    print("  применяется: x_norm = (x - mu)/sigma на реальных пакетах; dir не трогаем")


def main() -> None:
    """Точка входа: открыть ``splits.npz`` и напечатать полный осмотр."""
    ap = argparse.ArgumentParser(description="Осмотр бандла сплита splits.npz")
    ap.add_argument("--splits", default="../splits/splits.npz", help="путь к splits.npz")
    args = ap.parse_args()

    z = np.load(args.splits, allow_pickle=False)
    dump_keys(z)
    days = collect_days(z)
    print(f"Дней в сплите: {len(days)} -> {days}\n")
    report_sizes(z, days)
    report_coverage(z, days)
    report_norm(z)


if __name__ == "__main__":
    main()
