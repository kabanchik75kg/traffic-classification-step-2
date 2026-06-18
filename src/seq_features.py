#!/usr/bin/env python3
"""seq_features.py — общее ядро sequence-представления.

Этот модуль содержит датасет-независимые примитивы, из которых строится
тензорное представление потока φ_seq: S → ℝ^{N×3} = [size, dir, IAT].

Зачем выделен в отдельный файл.
    Тензоризация обучающего набора (CSE-CIC-IDS2018) и тестового набора
    (CIC-IDS2017) ДОЛЖНА быть идентичной до последнего бита — иначе
    zero-shot перенос модели 2018 → 2017 перестаёт быть корректным
    сравнением. Раньше функции `build_features`, `col_to_matrix` и т.д.
    были скопированы из одного скрипта в другой «байт-в-байт»; любая
    правка одной копии незаметно расходила представления двух датасетов.
    Теперь ядро живёт в одном месте, а оба тензоризатора его импортируют —
    идентичность гарантирована структурно, а не дисциплиной.

Что НЕ входит в это ядро.
    Нормализация (z-score размера и log-IAT) здесь сознательно отсутствует.
    Тензоры хранят «сырое» представление: size — реальный размер пакета,
    iat — log1p(межпакетный интервал), dir — {−1, 0, +1}. Параметры
    нормализации μ/σ оцениваются ТОЛЬКО по обучающей выборке на этапе
    обучения и затем замораживаются (в т.ч. для 2017). Поэтому нормализация —
    артефакт модели, а не представления, и в этом модуле её быть не должно.

Соглашение о кодировке.
    PAD = -1 — маркер «нет пакета» во всех трёх каналах после паддинга.
    Маска реальных пакетов вычисляется как (size != PAD): размер пакета
    всегда ≥ 0, поэтому −1 однозначно отличает паддинг от данных.
    Направление: NFStream отдаёт splt_dir ∈ {0, 1} (0 = от инициатора
    потока, т.е. первый пакет всегда 0), здесь оно ремапится в
    {0 → +1, 1 → −1, паддинг → 0}. Инвариант «первый пакет = +1» следует
    из определения потока в NFStream и проверяется на данных отдельно.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# --- Соглашения о кодировке и именах столбцов -------------------------------

PAD: int = -1
"""Маркер паддинга во всех каналах. Размер пакета ≥ 0, поэтому −1 безопасен."""

BENIGN_TOKENS: set[str] = {"", "benign", "normal", "background", "nan", "none", "0"}
"""Метки (в нижнем регистре), трактуемые как «не атака». Всё остальное — атака."""

LABEL_COL: str = "label"
SRCFILE_COL: str = "source_file"
NPKT_COL: str = "bidirectional_packets"
FIRST_COL: str = "first_seen_ms"

PS_COL_CANDIDATES: list[str] = ["splt_ps"]
DIR_COL_CANDIDATES: list[str] = ["splt_dir", "splt_direction"]
PIAT_COL_CANDIDATES: list[str] = ["splt_piat_ms", "splt_piat"]


# --- Разворачивание splt-массивов в матрицу [потоки × N] --------------------

def matrix_from_flat(
    flat: Any, n_rows: int, L: int, N: int, fill: int = PAD
) -> np.ndarray:
    """Плоский буфер длины ``n_rows * L`` → матрица ``[n_rows × N]``.

    Быстрый путь для случая, когда у всех потоков батча splt-массивы равной
    длины ``L``: тогда их можно развернуть одним ``reshape`` без Python-цикла.

    Parameters
    ----------
    flat : array-like
        Сцепленные значения всех потоков (результат ``ListArray.flatten``).
    n_rows : int
        Число потоков (строк) в батче.
    L : int
        Длина splt-массива каждого потока (одинаковая для всех в этом батче).
    N : int
        Целевая длина последовательности. При ``L >= N`` массив усекается,
        при ``L < N`` дополняется значением ``fill`` справа.
    fill : int, default PAD
        Значение паддинга для позиций за пределами реальных пакетов.

    Returns
    -------
    np.ndarray
        Матрица ``[n_rows × N]`` типа float64.
    """
    m = np.asarray(flat, dtype=np.float64).reshape(n_rows, L)
    if L >= N:
        return m[:, :N].copy()
    out = np.full((n_rows, N), fill, dtype=np.float64)
    out[:, :L] = m
    return out


def lists_to_matrix(seq: Sequence[Any], N: int, fill: int = PAD) -> np.ndarray:
    """Список splt-массивов разной длины → матрица ``[len(seq) × N]``.

    Медленный запасной путь (Python-цикл) для случая, когда массивы потоков
    имеют разную длину и быстрый ``matrix_from_flat`` неприменим. Каждый
    массив усекается до ``N`` или дополняется ``fill`` справа; ``None``
    трактуется как полностью паддинговая строка.

    Parameters
    ----------
    seq : sequence of array-like or None
        По одному splt-массиву на поток.
    N : int
        Целевая длина последовательности.
    fill : int, default PAD
        Значение паддинга.

    Returns
    -------
    np.ndarray
        Матрица ``[len(seq) × N]`` типа float64.
    """
    out = np.full((len(seq), N), fill, dtype=np.float64)
    for i, a in enumerate(seq):
        if a is None:
            continue
        a = np.asarray(a, dtype=np.float64).ravel()
        m = min(a.shape[0], N)
        if m:
            out[i, :m] = a[:m]
    return out


def col_to_matrix(arr: Any, N: int, fill: int = PAD) -> np.ndarray:
    """Столбец pyarrow ``ListArray`` → матрица ``[len × N]``.

    Выбирает быстрый путь (``matrix_from_flat``), когда у всех потоков
    одинаковая длина splt-массива и нет null-значений; иначе откатывается
    на медленный (``lists_to_matrix``). Результат идентичен в обоих случаях.

    Parameters
    ----------
    arr : pyarrow.ListArray
        Уже отобранный (``.take``) столбец splt-значений.
    N : int
        Целевая длина последовательности.
    fill : int, default PAD
        Значение паддинга.

    Returns
    -------
    np.ndarray
        Матрица ``[len(arr) × N]`` типа float64.
    """
    n = len(arr)
    if n == 0:
        return np.empty((0, N), dtype=np.float64)
    lens = arr.value_lengths().to_numpy(zero_copy_only=False)
    if arr.null_count == 0 and lens.min() == lens.max():
        L = int(lens[0])
        flat = arr.flatten().to_numpy(zero_copy_only=False)
        return matrix_from_flat(flat, n, L, N, fill)
    return lists_to_matrix(arr.to_pylist(), N, fill)


# --- Сборка признаков потока ------------------------------------------------

def build_features(
    ps_mat: np.ndarray, dir_mat: np.ndarray, piat_mat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Три splt-матрицы → тензор признаков ``X`` и маску реальных пакетов.

    Собирает попакетное представление потока из трёх каналов:

    * **size** — размер пакета; на паддинге обнуляется (0.0);
    * **dir**  — направление: ``0 → +1`` (от инициатора потока),
      ``1 → −1`` (к инициатору), паддинг → 0;
    * **iat**  — ``log1p`` от межпакетного интервала (предварительно
      обрезанного снизу нулём); на паддинге обнуляется (0.0).

    Маска вычисляется по каналу size: позиция реальна ⇔ ``ps != PAD``.
    Нормализация здесь НЕ применяется (см. докстринг модуля).

    Parameters
    ----------
    ps_mat : np.ndarray
        Матрица размеров пакетов ``[потоки × N]`` (паддинг = PAD).
    dir_mat : np.ndarray
        Матрица направлений ``[потоки × N]`` в кодировке NFStream {0, 1}
        (паддинг = PAD).
    piat_mat : np.ndarray
        Матрица межпакетных интервалов (мс) ``[потоки × N]`` (паддинг = PAD).

    Returns
    -------
    X : np.ndarray
        Тензор признаков ``[потоки × N × 3]`` типа float32
        в порядке каналов [size, dir, iat].
    mask : np.ndarray
        Маска реальных пакетов ``[потоки × N]`` типа uint8 (1 = пакет, 0 = паддинг).
    """
    mask = (ps_mat != PAD).astype(np.uint8)
    size = np.where(mask == 1, ps_mat, 0.0)
    direction = np.zeros_like(dir_mat, dtype=np.float64)
    direction[dir_mat == 0] = 1.0
    direction[dir_mat == 1] = -1.0
    iat = np.where(mask == 1, np.log1p(np.clip(piat_mat, 0, None)), 0.0)
    X = np.stack([size, direction, iat], axis=-1).astype(np.float32)
    return X, mask


# --- Метки и служебное ------------------------------------------------------

def is_malicious(labels: Any) -> np.ndarray:
    """Массив строковых меток → булева маска «атака».

    Метка считается benign, если её нормализованная форма (обрезка пробелов
    и нижний регистр) входит в :data:`BENIGN_TOKENS`; иначе — атака.

    Parameters
    ----------
    labels : array-like of str
        Сырые метки потоков (например, ``"Benign"``, ``"DoS Hulk"``, ``"benign"``).

    Returns
    -------
    np.ndarray
        Булев массив той же длины: True = атака, False = benign.
    """
    norm = np.array([str(x).strip().lower() for x in labels])
    return ~np.isin(norm, list(BENIGN_TOKENS))


def pick_col(available: Any, candidates: Sequence[str], what: str) -> str:
    """Выбрать первое присутствующее имя столбца из списка кандидатов.

    Делает тензоризаторы устойчивыми к мелким отличиям схемы parquet
    (например, ``splt_dir`` против ``splt_direction``).

    Parameters
    ----------
    available : collection of str
        Имена столбцов, имеющиеся в файле.
    candidates : sequence of str
        Кандидаты в порядке предпочтения.
    what : str
        Человекочитаемое имя столбца для сообщения об ошибке.

    Returns
    -------
    str
        Имя первого найденного столбца.

    Raises
    ------
    KeyError
        Если ни один кандидат не присутствует в ``available``.
    """
    for c in candidates:
        if c in available:
            return c
    raise KeyError(f"Не найден столбец {what}; есть: {sorted(available)}")
