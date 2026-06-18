#!/usr/bin/env python3
"""seq_models.py — реестр sequence-моделей над φ_seq = [size, dir, IAT].

Все модели имеют единый интерфейс ``forward(x, mask) -> logits``:
    * ``x``    — тензор [B, L, 3] (size, dir, IAT), нормализованный на лету;
    * ``mask`` — [B, L] float (1 = реальный пакет, 0 = паддинг);
    * выход    — логиты [B] (BCEWithLogitsLoss; sigmoid берётся снаружи).

Единый интерфейс позволяет :mod:`train_seq` и :mod:`infer_2017` работать с
любой архитектурой через :func:`make_model`, не зная её устройства, — добавить
модель в сравнение значит добавить класс и строку в реестр.

Покрытые семейства архитектур (ось сравнения внутри φ_seq):
    * свёртка       : :class:`CNN1D` (наивная), :class:`TCN` (дилатированная);
    * рекуррентность: :class:`BiLSTM`, :class:`GRUNet`;
    * внимание      : :class:`TransformerEnc`.

Маскинг. Свёрточные модели после свёрток применяют masked average pooling
(усреднение только по реальным пакетам). Рекуррентные используют
``pack_padded_sequence`` по реальным длинам. Трансформер исключает паддинг
через ``src_key_padding_mask`` и усредняет только реальные позиции. Каждый
поток содержит ≥1 реальный пакет (первый пакет всегда реален), поэтому
деления на нулевую длину не возникает.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn


def masked_avg_pool_cl(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Среднее по реальным позициям для тензора формы [B, C, L].

    Parameters
    ----------
    h : torch.Tensor
        Карта признаков [B, C, L] (канал-первый, как после Conv1d).
    mask : torch.Tensor
        Маска [B, L] (1 = пакет, 0 = паддинг).

    Returns
    -------
    torch.Tensor
        Пулинг [B, C]; деление на ``clamp(min=1)`` страхует от пустых строк.
    """
    m = mask.unsqueeze(1)  # [B, 1, L]
    return (h * m).sum(2) / m.sum(2).clamp(min=1)


# --- Свёрточные модели ------------------------------------------------------

class CNN1D(nn.Module):
    """Наивная 1D-CNN: три свёрточных блока + masked average pooling.

    Перенесена дословно из исходного ``train_seq.py`` (ядро 5, каналы
    32→64→64, BatchNorm+ReLU), чтобы переобучение воспроводило прежнее
    поведение модели.
    """

    def __init__(self, in_ch: int = 3, hidden=(32, 64, 64), k: int = 5, p: float = 0.3):
        super().__init__()
        layers, c = [], in_ch
        for h in hidden:
            layers += [nn.Conv1d(c, h, k, padding=k // 2), nn.BatchNorm1d(h), nn.ReLU()]
            c = h
        self.conv = nn.Sequential(*layers)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(c, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2))          # [B, C, L]
        pooled = masked_avg_pool_cl(h, mask)
        return self.fc(self.drop(pooled)).squeeze(1)


class _Chomp1d(nn.Module):
    """Срезает лишний правый паддинг после дилатированной свёртки (причинность)."""

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp].contiguous() if self.chomp > 0 else x


class _TCNBlock(nn.Module):
    """Остаточный блок TCN: две причинные дилатированные свёртки + skip-связь."""

    def __init__(self, c_in: int, c_out: int, k: int, dilation: int, p: float):
        super().__init__()
        pad = (k - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation), _Chomp1d(pad),
            nn.BatchNorm1d(c_out), nn.ReLU(), nn.Dropout(p),
            nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation), _Chomp1d(pad),
            nn.BatchNorm1d(c_out), nn.ReLU(), nn.Dropout(p),
        )
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """Temporal Convolutional Network: стек причинных дилатированных свёрток.

    Дилатации (1, 2, 4, 8, 16) при ядре 3 дают рецептивное поле, покрывающее
    всю последовательность N=60. «Правильная» свёрточная архитектура —
    контроль к наивной :class:`CNN1D`: показывает, дело ли в свёрточном
    семействе как таковом или в наивности конкретной CNN.
    """

    def __init__(self, in_ch: int = 3, hidden: int = 64, k: int = 3,
                 dilations=(1, 2, 4, 8, 16), p: float = 0.3):
        super().__init__()
        blocks, c = [], in_ch
        for d in dilations:
            blocks.append(_TCNBlock(c, hidden, k, d, p))
            c = hidden
        self.tcn = nn.Sequential(*blocks)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.tcn(x.transpose(1, 2))           # [B, C, L]
        pooled = masked_avg_pool_cl(h, mask)
        return self.fc(self.drop(pooled)).squeeze(1)


# --- Рекуррентные модели ----------------------------------------------------

class BiLSTM(nn.Module):
    """Двунаправленный LSTM по реальным длинам потока (pack_padded).

    Перенесён дословно из исходного ``train_seq.py``. Решение строится по
    конкатенации финальных скрытых состояний обоих направлений.
    """

    def __init__(self, in_ch: int = 3, hidden: int = 64, p: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.long().sum(1).cpu()
        packed = rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)            # hn [2, B, H]
        h = torch.cat([hn[0], hn[1]], dim=1)      # [B, 2H]
        return self.fc(self.drop(h)).squeeze(1)


class GRUNet(nn.Module):
    """Двунаправленный GRU — лёгкий рекуррентный контроль к :class:`BiLSTM`.

    Та же схема (pack_padded + конкатенация финальных состояний), но GRU имеет
    меньше параметров и не хранит cell-state: проверяет, что перенос —
    свойство рекуррентности, а не именно LSTM-гейтинга.
    """

    def __init__(self, in_ch: int = 3, hidden: int = 64, p: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(in_ch, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.long().sum(1).cpu()
        packed = rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, hn = self.gru(packed)                  # hn [2, B, H]
        h = torch.cat([hn[0], hn[1]], dim=1)      # [B, 2H]
        return self.fc(self.drop(h)).squeeze(1)


# --- Модель внимания --------------------------------------------------------

class _PositionalEncoding(nn.Module):
    """Синусоидальное позиционное кодирование (порядок пакетов значим)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerEnc(nn.Module):
    """Небольшой Transformer-энкодер с masked self-attention над пакетами.

    Проекция 3→d_model, синусоидальные позиции, ``num_layers`` слоёв
    self-attention с исключением паддинга (``src_key_padding_mask``), затем
    усреднение по реальным позициям и линейная голова. Покрывает семейство
    «внимание», отсутствовавшее в исходном сравнении.
    """

    def __init__(self, in_ch: int = 3, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, ff: int = 128, p: float = 0.3):
        super().__init__()
        self.proj = nn.Linear(in_ch, d_model)
        self.pos = _PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=ff,
                                           dropout=p, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.pos(self.proj(x))                       # [B, L, d_model]
        key_padding = mask == 0                          # True = паддинг (игнор)
        h = self.enc(h, src_key_padding_mask=key_padding)
        m = mask.unsqueeze(-1)                            # [B, L, 1]
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)  # masked mean -> [B, d_model]
        return self.fc(self.drop(pooled)).squeeze(1)


# --- Реестр -----------------------------------------------------------------

_REGISTRY = {
    "cnn": CNN1D,
    "tcn": TCN,
    "bilstm": BiLSTM,
    "gru": GRUNet,
    "transformer": TransformerEnc,
}

MODEL_NAMES: list[str] = list(_REGISTRY)


def make_model(name: str, seq_len: int) -> nn.Module:
    """Построить модель по имени.

    Parameters
    ----------
    name : str
        Одно из :data:`MODEL_NAMES` (cnn / tcn / bilstm / gru / transformer).
    seq_len : int
        Длина последовательности. Архитектуры самонастраиваются под длину
        (свёртки — padding, RNN — pack_padded, трансформер — позиции до 512),
        параметр сохранён для совместимости интерфейса и будущих моделей.

    Returns
    -------
    torch.nn.Module
        Модель с интерфейсом ``forward(x, mask) -> logits[B]``.

    Raises
    ------
    ValueError
        Если имя модели не зарегистрировано.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"неизвестная модель {name!r}; доступны: {MODEL_NAMES}")
    return _REGISTRY[key]()
