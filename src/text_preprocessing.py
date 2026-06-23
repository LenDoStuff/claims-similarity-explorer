from __future__ import annotations

import re
from typing import Any

import pandas as pd


WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return WHITESPACE_RE.sub(" ", str(value).replace("\x00", " ")).strip()
