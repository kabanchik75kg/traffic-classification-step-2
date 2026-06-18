#!/usr/bin/env python3
"""
Проверка тензоров после шага 2.2 — по всем дням сразу.

Для каждого .npz проверяет инварианты представления, затем делает
кросс-файловые проверки (одинаковый N, форма, dtype), сводит классы по дням
и печатает сигнатуры классов (пример потока). Memory-safe: файлы грузятся
по одному.

Запуск:
    python verify_tensors.py --dir tensors             # все дни
    python verify_tensors.py --file tensors/14-02.npz  # один файл

Зависит только от numpy.
"""
import argparse
import glob
import os
import numpy as np

PAD = -1


def invariants(X, mask, y, N):
    size, direction, iat = X[..., 0], X[..., 1], X[..., 2]
    lengths = mask.sum(axis=1)
    return [
        ("len(X)==len(mask)==len(y)", len(X) == len(mask) == len(y)),
        ("маска — префикс (нет дыр)", bool((mask[:, :-1] >= mask[:, 1:]).all())),
        ("у каждого потока >=1 пакет", bool((mask[:, 0] == 1).all())),
        ("dir in {-1,0,+1}", bool(np.isin(direction, [-1.0, 0.0, 1.0]).all())),
        ("dir==0 <=> mask==0", bool(((direction == 0) == (mask == 0)).all())),
        ("первый пакет dir==+1", bool((direction[:, 0] == 1.0).all())),
        ("size>0 где mask==1", bool((size[mask == 1] > 0).all())),
        ("size==0 где mask==0", bool((size[mask == 0] == 0).all())),
        ("iat>=0 везде", bool((iat >= 0).all())),
        ("iat==0 где mask==0", bool((iat[mask == 0] == 0).all())),
        ("нет NaN/inf в X", bool(np.isfinite(X).all())),
        (f"длина по маске <= N({N})", bool((lengths <= N).all())),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="tensors")
    ap.add_argument("--file", default=None)
    args = ap.parse_args()

    files = [args.file] if args.file else sorted(glob.glob(os.path.join(args.dir, "*.npz")))
    if not files:
        raise SystemExit(f"Нет .npz в {args.dir}")
    print(f"Проверка тензоров ({len(files)} файлов)\n")

    all_ok = True
    Ns, feat_dims, dtypes = set(), set(), set()
    tot, tot_mal = 0, 0
    class_total, class_days, class_example = {}, {}, {}

    print("ПО ФАЙЛАМ:")
    for f in files:
        z = np.load(f, allow_pickle=False)
        X, mask, y = z["X"], z["mask"], z["y"]
        classes, codes = z["label_classes"], z["label_codes"]
        day, N = str(z["day"]), int(z["N"])

        res = invariants(X, mask, y, N)
        fails = [name for name, ok in res if not ok]
        ok = len(fails) == 0
        all_ok &= ok
        flag = f"{len(res)}/{len(res)} OK" if ok else f"НАРУШЕНИЯ: {', '.join(fails)}"
        print(f"  {day}: [{flag}] потоков {len(y):,} | атак {int(y.sum()):,} | N={N}")

        Ns.add(N); feat_dims.add(X.shape[2]); dtypes.add(str(X.dtype))
        tot += len(y); tot_mal += int(y.sum())

        size, direction, iat = X[..., 0], X[..., 1], X[..., 2]
        for i, c in enumerate(classes):
            m = codes == i
            n = int(m.sum())
            class_total[c] = class_total.get(c, 0) + n
            class_days.setdefault(c, set()).add(day)
            if c not in class_example and n > 0:
                r = np.where(m)[0][0]
                L = int(mask[r].sum()); k = min(L, 6)
                class_example[c] = (L, size[r, :k].astype(int).tolist(),
                                    direction[r, :k].astype(int).tolist(),
                                    np.round(iat[r, :k], 2).tolist())

    print("\nКРОСС-ФАЙЛОВЫЕ ПРОВЕРКИ:")
    for name, cond in [
        (f"N одинаков везде ({sorted(Ns)})", len(Ns) == 1),
        (f"признаков на пакет = 3 ({sorted(feat_dims)})", feat_dims == {3}),
        (f"dtype X = float32 ({sorted(dtypes)})", dtypes == {"float32"}),
    ]:
        all_ok &= cond
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}")

    print("\nСВОДКА ПО КЛАССАМ (всего по всем дням):")
    for c in sorted(class_total, key=lambda k: -class_total[k]):
        days = class_days[c]
        days_str = "все дни" if len(days) >= len(files) else ", ".join(sorted(days))
        print(f"  {c:<20} {class_total[c]:>12,}   [{days_str}]")
    print(f"  {'-'*48}")
    print(f"  ИТОГО потоков {tot:,}, атак {tot_mal:,} ({100*tot_mal/max(tot,1):.1f}%)")

    print("\nСИГНАТУРЫ КЛАССОВ (пример потока, первые 6 пакетов; dir +1=к жертве):")
    for c in sorted(class_example):
        L, sizes, dirs, iats = class_example[c]
        print(f"  {c:<20} len={L:<5} size={sizes} dir={dirs} log1p_iat={iats}")

    print("\n" + ("ВСЁ ЧИСТО ✅" if all_ok else "ЕСТЬ НАРУШЕНИЯ — разбираемся ❌"))


if __name__ == "__main__":
    main()