from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


def transpose_virials(data: dict[str, Any]) -> dict[str, Any]:
    """Switch between dpdata's position-first and DeePMD's force-first virials."""
    if "virials" not in data:
        return data
    return {**data, "virials": data["virials"].swapaxes(-1, -2)}
