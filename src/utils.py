"""
Logging helpers and miscellaneous utilities.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import List


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s – %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def natural_sort_key(text: str):
    """
    Key function for natural (human-friendly) sorting.

    '0001.tdms' < '0002.tdms' < '0010.tdms' instead of lexicographic order.
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def sorted_naturally(paths: List[str]) -> List[str]:
    """Sort a list of paths using natural sort."""
    return sorted(paths, key=natural_sort_key)
