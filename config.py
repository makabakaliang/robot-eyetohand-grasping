"""Runtime configuration loader with defaults and validation."""
import json
from pathlib import Path


class RuntimeConfig:
    """Typed wrapper around runtime_config.json."""

    DEFAULTS = {
        "manual_comp": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "tool_comp_enabled": True,
        "tool_offset_xyz": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "soft_limit_enabled": False,
        "z_min_limit": 0.0,
    }

    def __init__(self, path="runtime_config.json"):
        self.path = Path(path)
        self._data = dict(self.DEFAULTS)
        if self.path.exists():
            self._data.update(json.loads(self.path.read_text(encoding="utf-8")))
        self._validate()

    def _validate(self):
        if len(self._data["manual_comp"]) != 6:
            raise ValueError("manual_comp must have 6 values")
        if len(self._data["tool_offset_xyz"]) != 3:
            raise ValueError("tool_offset_xyz must have 3 values")

    @property
    def tool_offset(self):
        return tuple(self._data["tool_offset_xyz"])

    @property
    def z_min(self):
        return float(self._data["z_min_limit"])

    @property
    def soft_limit_enabled(self):
        return bool(self._data["soft_limit_enabled"])

    def manual_compensation(self):
        return list(self._data["manual_comp"])
