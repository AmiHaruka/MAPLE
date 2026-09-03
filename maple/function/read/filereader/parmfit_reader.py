"""Parmfit external key/value configuration reader."""

from __future__ import annotations

import os
import re
from typing import Any, Optional


_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _auto_cast(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        pass
    return value


def _strip_inline_comment(line: str) -> str:
    text = line.strip()
    if not text or text.startswith(("#", ";")):
        return ""
    for index, char in enumerate(text):
        if char in {"#", ";"} and (index == 0 or text[index - 1].isspace()):
            return text[:index].rstrip()
    return text


def _resolve_file_path(path: str, base_dir: str, *, label: str) -> str:
    text = str(path).strip()
    if not text:
        return ""
    resolved = text if os.path.isabs(text) else os.path.join(base_dir, text)
    resolved = os.path.abspath(resolved)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"parmfit config file not found for '{label}': {path}")
    return resolved


class ParmfitReader:
    """Read a parmfit-only key=value file and return normalized params."""

    FORBIDDEN_KEYS = {
        "input",
        "pdb",
        "model",
        "model_options",
        "device",
        "pbc",
        "gpuid",
        "d4",
        "task",
        "sp",
        "opt",
        "ts",
        "scan",
        "freq",
        "irc",
        "md",
        "solv",
        "ensemble",
    }
    PATH_KEYS = {"mol2"}
    MULTI_PATH_KEYS = {"cfmol2"}

    def __new__(cls, file_path: str, base_dir: Optional[str] = None) -> dict[str, Any]:
        resolved = cls.resolve_path(file_path, base_dir=base_dir)
        return cls._read_params(resolved)

    @staticmethod
    def resolve_path(file_path: str, base_dir: Optional[str] = None) -> str:
        text = str(file_path).strip()
        if not text:
            raise ValueError("parmfit(input=...) requires a config file path.")
        resolved = text if os.path.isabs(text) else os.path.join(base_dir if base_dir is not None else os.getcwd(), text)
        resolved = os.path.abspath(resolved)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"parmfit config file not found: {file_path}")
        return resolved

    @classmethod
    def _read_params(cls, path: str) -> dict[str, Any]:
        params: dict[str, Any] = {"method": "abinitio"}
        seen: set[str] = set()
        base_dir = os.path.dirname(os.path.abspath(path))

        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for lineno, raw in enumerate(handle, start=1):
                line = _strip_inline_comment(raw)
                if not line:
                    continue
                if "=" not in line:
                    raise ValueError(f"Invalid parmfit config line {path}:{lineno}; expected key=value.")
                key, value = line.split("=", 1)
                key = key.strip().lower()
                value = value.strip()
                if not _KEY_RE.match(key):
                    raise ValueError(f"Invalid parmfit config key {key!r} at {path}:{lineno}.")
                if key in seen:
                    raise ValueError(f"Duplicate parmfit config key {key!r} at {path}:{lineno}.")
                seen.add(key)
                if key == "pdb":
                    raise ValueError("Do not set 'pdb=' in parmfit input config. Provide the structure as a PDB <path> block.")
                if key == "input":
                    raise ValueError("Do not set 'input=' inside a parmfit input config.")
                if key == "frcmod":
                    raise ValueError("Do not set 'frcmod=' in parmfit input config. Correction frcmod files are generated automatically.")
                if key in cls.FORBIDDEN_KEYS:
                    raise ValueError(f"Parmfit input config only accepts parmfit parameters; unsupported global key {key!r}.")

                if key in cls.PATH_KEYS:
                    params[key] = _resolve_file_path(value, base_dir, label=key)
                elif key in cls.MULTI_PATH_KEYS:
                    entries = value.replace(",", " ").split()
                    params[key] = " ".join(_resolve_file_path(entry, base_dir, label=key) for entry in entries)
                else:
                    params[key] = _auto_cast(value)

        return params
