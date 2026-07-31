"""Adapter that loads the frozen SAP engine snapshot used by the thesis env."""

from __future__ import annotations

import sys
from pathlib import Path


def _vendor_root() -> Path:
    # sap_thesis_env/adapters/sapai_backend.py -> sap_thesis_env/vendor/sapai_frozen
    return Path(__file__).resolve().parents[1] / "vendor" / "sapai_frozen"


def ensure_sapai_importable() -> Path:
    root = _vendor_root()
    if not root.exists():
        raise RuntimeError(
            f"Frozen SAP backend not found at {root}. "
            "Expected vendor snapshot in sap_thesis_env/vendor/sapai_frozen."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def load_sapai_symbols() -> dict[str, object]:
    """Return the SAP symbols required by the frozen arena environment."""
    ensure_sapai_importable()
    from sapai import Battle, Player, Team, data  # type: ignore
    from sapai.shop import get_shop_rules  # type: ignore

    return {
        "Player": Player,
        "Team": Team,
        "Battle": Battle,
        "data": data,
        "get_shop_rules": get_shop_rules,
    }
