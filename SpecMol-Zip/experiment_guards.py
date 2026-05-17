"""Small runtime guards for experiment reproducibility.

These checks are intentionally lightweight and import-safe.  They prevent
accidentally launching BACE runs on pre-B4 processed data directories.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union


LEGACY_BACE_DATA_ROOTS = {
    "down_task",
    "down_task_2d",
    "down_task_v2",
    "down_task_unifold",
    "down_task_bondpair",
    "down_task_unimol",
    "down_task_unimolupd",
    "down_task_ctrl",
}


def assert_not_legacy_bace_path(path: Union[str, os.PathLike], *, context: str) -> None:
    """Raise if a BACE run points at a known pre-B4 archive directory.

    B4 changed chemical bond graph construction from one-way to bidirectional
    edges. Existing processed BACE .pt files under legacy roots were generated
    before that fix and must not be used for canonical V0/V1/V2 comparisons.
    """
    root_name = Path(path).name
    if root_name not in LEGACY_BACE_DATA_ROOTS:
        return
    raise RuntimeError(
        f"{context} points at '{root_name}', a pre-B4 BACE archive. "
        "Regenerate and use 'down_task_bace_v2' for V2/T6/T7-style data or "
        "'down_task_bace_unifold' for V0 baseline data. "
        "Do not compare results from the legacy processed .pt files."
    )


def require_explicit_experimental_run(flag_value: bool, *, experiment: str) -> None:
    """Require an explicit opt-in for paused exploratory experiment lines."""
    if flag_value:
        return
    raise RuntimeError(
        f"{experiment} is paused for the rescue roadmap. "
        "First complete the BACE post-B4 canonical V0/V2-T5/T6 table. "
        f"Pass the script's explicit allow flag only for a deliberate {experiment} audit."
    )
