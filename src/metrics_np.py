#!/usr/bin/env python3
"""metrics_np.py — метрики оценки этапа 1, общие для seq- и agg-моделей.

Выделены в отдельный модуль (без torch), чтобы и :mod:`evaluate` (seq-модели),
и :mod:`infer_2017_baseline` (XGBoost на φ_agg) считали метрики ОДНИМ кодом —
иначе строки «seq» и «agg» в одной таблице оказались бы посчитаны по-разному.

Состав блока (как в Таблице 4.1 / 5.1 этапа 1):
    MCC, F1-macro, PR-AUC, ROC-AUC, FPR при TPR=0,95.

Соглашение о порогах:
    * argmax-F1 на валидации 2018 — операционный порог этапа 1 (фиксируется и
      применяется без подстройки к test-2018 и к 2017);
    * максимум MCC по всем порогам (оракул) — режим «оптимальный порог» этапа 1,
      отделяющий качество ранжирования от переносимости калибровки.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score, roc_curve)


def best_f1_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Порог, максимизирующий F1 положительного класса (правило порога этапа 1).

    Считается по PR-кривой валидации; возвращается порог из точки с наибольшим
    F1. На вырожденных входах (один класс / одинаковые скоры) — 0.5.
    """
    if len(np.unique(y)) < 2:
        return 0.5
    p, r, thr = precision_recall_curve(y, scores)
    if len(thr) == 0:
        return 0.5
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)   # длина = len(thr)+1
    i = int(np.nanargmax(f1[:-1]))                  # последняя точка без порога
    return float(thr[i])


def best_mcc(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Максимальный MCC по всем порогам (оракул) и порог в этой точке.

    Векторно: сортируем по убыванию скора и для каждого среза «верхние k —
    атака» считаем MCC из 2×2-таблицы. Это режим «оптимальный порог» этапа 1 —
    он отделяет качество ранжирования от переносимости калибровки.
    """
    y = y.astype(np.int64)
    P = int(y.sum()); Nn = len(y) - P
    if P == 0 or Nn == 0:
        return float("nan"), 0.5
    order = np.argsort(-scores, kind="mergesort")
    ys, ss = y[order], scores[order]
    tp = np.cumsum(ys).astype(np.float64)
    k = np.arange(1, len(y) + 1, dtype=np.float64)
    fp = k - tp
    fn = P - tp
    tn = Nn - fp
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    num = tp * tn - fp * fn
    mcc = np.where(denom > 0, num / np.where(denom > 0, denom, 1.0), 0.0)
    i = int(np.argmax(mcc))
    return float(mcc[i]), float(ss[i])


def fpr_at_tpr(scores: np.ndarray, y: np.ndarray, target_tpr: float = 0.95) -> float:
    """FPR в точке ROC-кривой, где TPR впервые достигает target_tpr."""
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, scores)
    idx = int(np.searchsorted(tpr, target_tpr))
    idx = min(idx, len(fpr) - 1)
    return float(fpr[idx])


def metric_block(scores: np.ndarray, y: np.ndarray, tau: float) -> dict:
    """Блок метрик этапа 1 при фиксированном пороге τ.

    Пороговозависимые (MCC, F1-macro) считаются при τ; пороговонезависимые
    (PR-AUC, ROC-AUC, FPR@TPR=0,95) — по скорам напрямую.
    """
    pred = (scores >= tau).astype(int)
    two_classes = len(np.unique(y)) == 2
    return {
        "MCC": matthews_corrcoef(y, pred) if two_classes else float("nan"),
        "F1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "PR_AUC": average_precision_score(y, scores) if y.any() else float("nan"),
        "ROC_AUC": roc_auc_score(y, scores) if two_classes else float("nan"),
        "FPR_at_TPR95": fpr_at_tpr(scores, y, 0.95),
    }