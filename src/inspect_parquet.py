"""
Осмотр размеченных parquet перед тензоризацией (шаг 2.2).
Печатает по каждому дню: число строк, схему, распределение меток (label)
и первые 15 строк в читабельном виде (splt-массивы показываются компактно).
 
Запуск в VS Code из корня проекта:
    python inspect_parquet.py
    python inspect_parquet.py --data data --n 15
 
Зависит от: pandas, pyarrow (уже стоят, ими писался parquet).
Memory-safe: для превью читаются только первые n строк, для меток — один
лёгкий проход по столбцу label. Полные 6 млн строк в RAM не грузятся.
"""
import argparse
import glob
import os
 
import pandas as pd
 
 
def fmt_arr(v, head=8):
    """Компактный показ splt-массивов: [120, 60, 1500, ...](n=37)."""
    try:
        seq = list(v)
    except TypeError:
        return v
    n = len(seq)
    shown = ", ".join(str(x) for x in seq[:head])
    tail = ", ..." if n > head else ""
    return f"[{shown}{tail}](n={n})"
 
 
def compact(df):
    """Копия df, где столбцы-массивы заменены на компактные строки."""
    out = df.copy()
    for c in out.columns:
        first = next((x for x in out[c] if x is not None), None)
        if isinstance(first, (list, tuple)) or hasattr(first, "__len__") and not isinstance(first, (str, bytes)):
            out[c] = out[c].map(fmt_arr)
    return out
 
 
def label_counts(path, label_col="label"):
    """Распределение меток одним лёгким проходом."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    if label_col not in pf.schema_arrow.names:
        return None
    counts = {}
    for batch in pf.iter_batches(batch_size=500_000, columns=[label_col]):
        s = batch.column(0).to_pandas().astype("string").fillna("<NaN>")
        for k, v in s.value_counts().items():
            counts[k] = counts.get(k, 0) + int(v)
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="папка с parquet")
    ap.add_argument("--n", type=int, default=15, help="строк превью на файл")
    args = ap.parse_args()
 
    import pyarrow.parquet as pq
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 42)
 
    files = sorted(glob.glob(os.path.join(args.data, "*.parquet")))
    if not files:
        raise SystemExit(f"Нет parquet в {args.data}")
    print(f"Найдено файлов: {len(files)}\n")
 
    for path in files:
        pf = pq.ParquetFile(path)
        nrows = pf.metadata.num_rows
        print("=" * 100)
        print(f"ФАЙЛ: {os.path.basename(path)}   строк: {nrows:,}   столбцов: {len(pf.schema_arrow.names)}")
        print("-" * 100)
 
        # схема и типы
        print("СТОЛБЦЫ И ТИПЫ:")
        for field in pf.schema_arrow:
            print(f"   {field.name:<22} {field.type}")
 
        # распределение меток
        lc = label_counts(path)
        if lc is not None:
            mal = sum(v for k, v in lc.items() if str(k).strip().lower()
                      not in {"", "benign", "normal", "background", "<nan>", "nan", "none", "0"})
            ben = nrows - mal
            print(f"\nМЕТКИ (label):  benign≈{ben:,}  atака≈{mal:,}  доля атак={100*mal/max(nrows,1):.3f}%")
            for k, v in lc.items():
                print(f"   {str(k):<22} {v:,}")
 
        # превью первых n строк
        batch = next(pf.iter_batches(batch_size=args.n))
        df = batch.to_pandas()
        print(f"\nПЕРВЫЕ {len(df)} СТРОК:")
        print(compact(df).to_string(index=False))
        print()
 
 
if __name__ == "__main__":
    main()