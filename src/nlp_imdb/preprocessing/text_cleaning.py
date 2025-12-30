# src/nlp_imdb/preprocessing/text_cleaning.py
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import Literal

Profile = Literal["classic", "rnn", "bert"]

_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class TextCleaningConfig:
    profile: Profile = "classic"
    strip_html: bool = True
    normalize_unicode: bool = True
    normalize_whitespace: bool = True
    lowercase: bool = True
    normalize_quotes_dashes: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_typography(s: str) -> str:
    # comillas “ ” ‘ ’ → " '
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("‘", "'").replace("’", "'")
    # guiones/en-dash/em-dash → -
    s = s.replace("–", "-").replace("—", "-")
    return s


def get_default_config(profile: Profile = "classic") -> TextCleaningConfig:
    # Por ahora mantenemos lo mismo para los 3 perfiles (seguro y consistente).
    return TextCleaningConfig(profile=profile)


def clean_text(text: str, cfg: TextCleaningConfig | None = None) -> str:
    if cfg is None:
        cfg = get_default_config("classic")

    s = text

    if cfg.strip_html:
        s = _HTML_RE.sub(" ", s)

    if cfg.normalize_unicode:
        s = unicodedata.normalize("NFKC", s)

    if cfg.normalize_quotes_dashes:
        s = _normalize_typography(s)

    if cfg.lowercase:
        s = s.lower()

    if cfg.normalize_whitespace:
        s = _WS_RE.sub(" ", s).strip()

    return s
