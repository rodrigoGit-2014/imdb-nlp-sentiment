# src/nlp_imdb/preprocessing/tokenization.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

from nlp_imdb.preprocessing.text_cleaning import (
    Profile,
    TextCleaningConfig,
    clean_text,
    get_default_config,
)

TokenizationProfile = Literal["classic", "rnn", "bert"]

# Tokenización simple (para RNN con embeddings estáticos)
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\s]")


@dataclass(frozen=True)
class RnnTokenizationConfig:
    """
    Config para tokenización/encoding de modelo secuencial (RNN/LSTM/GRU).

    - max_vocab: tamaño máximo del vocabulario (por frecuencia)
    - min_freq: frecuencia mínima para incluir token en vocab
    - max_length: longitud máxima de secuencia (padding/trunc)
    """

    max_vocab: int = 40_000
    min_freq: int = 2
    max_length: int = 256
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"


@dataclass(frozen=True)
class BertTokenizationConfig:
    """
    Config para tokenización en modelos contextualizados (BERT/RoBERTa/etc.)
    """

    model_name: str = "bert-base-uncased"
    max_length: int = 256


def clean_texts(
    texts: Iterable[str],
    profile: Profile,
    cfg: TextCleaningConfig | None = None,
) -> list[str]:
    """
    Limpieza batch (reutilizable por los 3 modelos).
    """
    cleaning_cfg = cfg or get_default_config(profile)
    return [clean_text(t, cleaning_cfg) for t in texts]


# -----------------------------
# Classic: TF-IDF / BoW
# -----------------------------
def classic_prepare_texts(
    texts: Iterable[str],
    cleaning_cfg: TextCleaningConfig | None = None,
) -> list[str]:
    """
    Devuelve textos listos para vectorizadores clásicos (TF-IDF/BoW).
    La tokenización final la hace scikit-learn internamente.
    """
    return clean_texts(texts, profile="classic", cfg=cleaning_cfg)


# -----------------------------
# RNN: vocab + encoding (ids)
# -----------------------------
def rnn_tokenize(text: str) -> list[str]:
    """
    Tokenizador simple y determinista para RNN (no depende de librerías externas).
    """
    return _WORD_RE.findall(text)


def build_vocab(
    texts: Iterable[str],
    cfg: RnnTokenizationConfig,
    cleaning_cfg: TextCleaningConfig | None = None,
) -> dict[str, int]:
    """
    Construye vocabulario por frecuencia. Devuelve dict token->id.

    ids reservados:
      0: PAD
      1: UNK
    """
    cleaned = clean_texts(texts, profile="rnn", cfg=cleaning_cfg)

    freq: dict[str, int] = {}
    for t in cleaned:
        for tok in rnn_tokenize(t):
            freq[tok] = freq.get(tok, 0) + 1

    # Orden por frecuencia desc, luego alfabético para determinismo
    items = [(tok, c) for tok, c in freq.items() if c >= cfg.min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))

    # Reservados
    vocab: dict[str, int] = {cfg.pad_token: 0, cfg.unk_token: 1}

    # Agregar hasta max_vocab
    # (max_vocab incluye PAD/UNK)
    limit = max(cfg.max_vocab - len(vocab), 0)
    for tok, _ in items[:limit]:
        if tok not in vocab:
            vocab[tok] = len(vocab)

    return vocab


def encode_rnn(
    texts: Iterable[str],
    vocab: dict[str, int],
    cfg: RnnTokenizationConfig,
    cleaning_cfg: TextCleaningConfig | None = None,
) -> list[list[int]]:
    """
    Convierte textos a secuencias de ids con padding/trunc a cfg.max_length.
    """
    cleaned = clean_texts(texts, profile="rnn", cfg=cleaning_cfg)
    unk_id = vocab.get(cfg.unk_token, 1)
    pad_id = vocab.get(cfg.pad_token, 0)

    encoded: list[list[int]] = []
    for t in cleaned:
        toks = rnn_tokenize(t)
        ids = [vocab.get(tok, unk_id) for tok in toks]

        # trunc
        if len(ids) > cfg.max_length:
            ids = ids[: cfg.max_length]

        # pad
        if len(ids) < cfg.max_length:
            ids = ids + [pad_id] * (cfg.max_length - len(ids))

        encoded.append(ids)

    return encoded


# -----------------------------
# BERT: HuggingFace tokenizer
# -----------------------------
def get_hf_tokenizer(model_name: str):
    """
    Carga tokenizer de HuggingFace (transformers).
    Import lazy para no obligar transformers en etapas que no lo usan.
    """
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "transformers no está instalado. Instálalo para usar BERT: pip install transformers"
        ) from e

    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def encode_bert(
    texts: Iterable[str],
    cfg: BertTokenizationConfig,
    cleaning_cfg: TextCleaningConfig | None = None,
) -> dict[str, list[list[int]]]:
    """
    Devuelve el dict típico de HF: input_ids, attention_mask (y token_type_ids si aplica).
    """
    tokenizer = get_hf_tokenizer(cfg.model_name)

    cleaned = clean_texts(texts, profile="bert", cfg=cleaning_cfg)
    batch = tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=cfg.max_length,
        return_attention_mask=True,
    )

    # Aseguramos tipos simples (listas) para mantenerlo serializable / fácil de inspeccionar
    out: dict[str, list[list[int]]] = {}
    for k, v in batch.items():
        out[k] = [list(row) for row in v]
    return out
