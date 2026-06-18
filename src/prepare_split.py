#!/usr/bin/env python3
"""
Шаг 2.3+2.4 ВКР — сплит без подбора и нормализация по train.

Делает стратифицированный СЛУЧАЙНЫЙ сплит 70/15/15 по классам (каждый класс,
включая микроклассы, представлен во всех трёх частях) и считает параметры
нормализации size/IAT ТОЛЬКО по обучающей выборке.

НЕ дублирует тензоры. Сохраняет компактно (один файл splits.npz):
  - split_<день>  : int8 на каждый поток (0=train, 1=val, 2=test)
  - class_<день>  : int16 глобальный id класса на каждый поток
  - classes       : список имён классов (глобальный словарь)
  - mu_size, sigma_size, mu_iat, sigma_iat : параметры нормализации (по train)
Нормализацию применяет обучающий скрипт на лету; те же μ,σ пойдут на 2017.

Запуск:
    python prepare_split.py --tensors tensors --out splits

Память: один день за раз (~1 ГБ). Зависит от numpy.
"""
import argparse
import glob
import os
import numpy as np

RATIOS = (0.70, 0.15, 0.15)  # train / val / test
BENIGN_TOKENS = {"benign", "normal", "background"}


def day_of(path):
    return os.path.splitext(os.path.basename(path))[0]


def stratified_3way(codes, seed, ratios=RATIOS):
    """codes: int-метки классов -> int8 [0/1/2] стратифицированно по классу."""
    rng = np.random.default_rng(seed)
    split = np.full(len(codes), -1, dtype=np.int8)
    for c in np.unique(codes):
        idx = np.where(codes == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(n * ratios[0]))
        n_va = int(round(n * ratios[1]))
        # гарантируем, что при n>=3 каждая часть непуста
        if n >= 3:
            n_tr = min(max(n_tr, 1), n - 2)
            n_va = min(max(n_va, 1), n - n_tr - 1)
        split[idx[:n_tr]] = 0
        split[idx[n_tr:n_tr + n_va]] = 1
        split[idx[n_tr + n_va:]] = 2
    assert (split >= 0).all()
    return split


class RunStats:
    """Потоковые среднее/СКО по выбранным значениям (Уэлфорд по чанкам не нужен —
    хватает сумм; значения ограничены, переполнения нет во float64)."""
    def __init__(self):
        self.n = 0
        self.s = 0.0
        self.ss = 0.0

    def update(self, v):
        v = np.asarray(v, dtype=np.float64)
        self.n += v.size
        self.s += v.sum()
        self.ss += (v * v).sum()

    def finalize(self):
        mu = self.s / self.n
        var = max(self.ss / self.n - mu * mu, 0.0)
        sigma = float(np.sqrt(var))
        return float(mu), (sigma if sigma > 1e-8 else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensors", default="tensors")
    ap.add_argument("--out", default="splits")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.tensors, "*.npz")))
    if not files:
        raise SystemExit(f"Нет .npz в {args.tensors}")
    os.makedirs(args.out, exist_ok=True)

    # глобальный словарь классов
    classes = set()
    for f in files:
        z = np.load(f, allow_pickle=False)
        classes.update([str(c) for c in z["label_classes"]])
    classes = sorted(classes)
    cls2gid = {c: i for i, c in enumerate(classes)}
    print(f"Классов всего: {len(classes)} -> {classes}\n")

    # сплит по дням + глобальные id классов; отчётные счётчики
    out = {}
    per = {0: {}, 1: {}, 2: {}}      # split -> {class -> count}
    ben_att = {0: [0, 0], 1: [0, 0], 2: [0, 0]}  # split -> [benign, attack]
    day_local = {}                    # день -> (split, gid) для прохода статистики
    for f in files:
        d = day_of(f)
        z = np.load(f, allow_pickle=False)
        local_classes = [str(c) for c in z["label_classes"]]
        local_codes = z["label_codes"]
        gid = np.array([cls2gid[local_classes[c]] for c in local_codes], dtype=np.int16)
        y = z["y"]
        split = stratified_3way(local_codes.astype(np.int64), args.seed)
        out[f"split_{d}"] = split
        out[f"class_{d}"] = gid
        day_local[d] = (split, gid, f)
        for s in (0, 1, 2):
            m = split == s
            for c in np.unique(gid[m]):
                per[s][classes[c]] = per[s].get(classes[c], 0) + int((gid[m] == c).sum())
            att = int(y[m].sum()); ben = int(m.sum()) - att
            ben_att[s][0] += ben; ben_att[s][1] += att

    # нормализация: статистика size/IAT по train, на реальных пакетах
    st_size, st_iat = RunStats(), RunStats()
    for d, (split, gid, f) in day_local.items():
        z = np.load(f, allow_pickle=False)
        X, mask = z["X"], z["mask"]
        tr = split == 0
        if not tr.any():
            continue
        mreal = mask[tr] == 1
        st_size.update(X[tr, :, 0][mreal])
        st_iat.update(X[tr, :, 2][mreal])
    mu_size, sigma_size = st_size.finalize()
    mu_iat, sigma_iat = st_iat.finalize()

    out.update(dict(
        classes=np.array(classes, dtype="U40"),
        mu_size=np.array(mu_size), sigma_size=np.array(sigma_size),
        mu_iat=np.array(mu_iat), sigma_iat=np.array(sigma_iat),
        seed=np.array(args.seed),
    ))
    out_path = os.path.join(args.out, "splits.npz")
    np.savez_compressed(out_path, **out)

    # ---------- отчёт ----------
    names = {0: "train", 1: "val", 2: "test"}
    tot = {s: ben_att[s][0] + ben_att[s][1] for s in (0, 1, 2)}
    grand = sum(tot.values())
    print("РАЗМЕРЫ СПЛИТОВ:")
    for s in (0, 1, 2):
        print(f"  {names[s]:<5} {tot[s]:>10,}  ({100*tot[s]/grand:.1f}%)  "
              f"benign {ben_att[s][0]:,} / атак {ben_att[s][1]:,}")
    print(f"  всего {grand:,}")

    print("\nПОКРЫТИЕ КЛАССОВ (train / val / test):")
    all_covered = True
    for c in classes:
        cnts = [per[s].get(c, 0) for s in (0, 1, 2)]
        ok = all(x > 0 for x in cnts)
        all_covered &= ok
        flag = "" if ok else "  <-- ПУСТО В КАКОЙ-ТО ЧАСТИ!"
        print(f"  {c:<20} {cnts[0]:>10,} / {cnts[1]:>8,} / {cnts[2]:>8,}{flag}")

    print(f"\nНОРМАЛИЗАЦИЯ (по train, реальные пакеты):")
    print(f"  size: mu={mu_size:.3f}  sigma={sigma_size:.3f}")
    print(f"  iat : mu={mu_iat:.3f}  sigma={sigma_iat:.3f}")
    print(f"  (применяется: x_norm = (x - mu)/sigma * mask; dir не трогаем)")

    print(f"\n-> {out_path}")
    print("ВСЕ КЛАССЫ ПОКРЫТЫ ✅" if all_covered else "ЕСТЬ ПУСТЫЕ КЛАССЫ ❌")


if __name__ == "__main__":
    main()