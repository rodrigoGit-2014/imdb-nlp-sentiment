# src/nlp_imdb/preprocessing/text_cleaning.py
from __future__ import annotations

import re
from dataclasses import dataclass


_WHITESPACE_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class CleanConfig:
    lowercase: bool = True
    strip_html: bool = True
    normalize_whitespace: bool = True


def clean_text(text: str, cfg: CleanConfig = CleanConfig()) -> str:
    """
    Minimal, safe text cleaning.
    - optionally strips HTML tags
    - optionally lowercases
    - optionally normalizes whitespace
    """
    out = text

    if cfg.strip_html:
        out = _HTML_TAG_RE.sub(" ", out)

    if cfg.lowercase:
        out = out.lower()

    if cfg.normalize_whitespace:
        out = _WHITESPACE_RE.sub(" ", out).strip()

    return out


def clean_corpus(texts: list[str], cfg: CleanConfig = CleanConfig()) -> list[str]:
    return [clean_text(t, cfg) for t in texts]
