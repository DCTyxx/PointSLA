#!/usr/bin/env python3
"""Compatibility launcher for point/text feature-space visualization.

Some server environments expose NumPy arrays for which ``torch.from_numpy``
raises ``TypeError: expected np.ndarray (got numpy.ndarray)`` because the
installed PyTorch/NumPy bridge is ABI-incompatible.  The Pointcept dataset
pipeline calls ``torch.from_numpy`` inside ``ToTensor`` before the visualization
script can extract any features.

This launcher patches only Pointcept's ``ToTensor`` class in this process.  It
prefers NumPy's DLPack interface, falls back to the Python buffer protocol, and
finally to a list conversion.  Training code and repository-wide transforms are
not modified.
"""

from __future__ import annotations

import os
import runpy
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pointcept.datasets.transform as pointcept_transform


def _dtype_for_buffer(dtype: np.dtype):
    dtype = np.dtype(dtype)
    mapping = {
        np.dtype(np.bool_): torch.bool,
        np.dtype(np.uint8): torch.uint8,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.int16): torch.int16,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
    }
    return mapping.get(dtype)


def _ndarray_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert NumPy -> Torch without relying on torch.from_numpy."""
    array = np.ascontiguousarray(array)

    # Preferred path: NumPy >= 1.22 implements the DLPack protocol.  This avoids
    # PyTorch's NumPy C-API bridge entirely and keeps the conversion efficient.
    if hasattr(array, "__dlpack__"):
        try:
            tensor = torch.utils.dlpack.from_dlpack(array)
            return tensor
        except Exception:
            pass

    # Second path: use Python's buffer protocol. Clone so the returned tensor
    # owns stable storage independent of the temporary NumPy array.
    torch_dtype = _dtype_for_buffer(array.dtype)
    if torch_dtype is not None:
        try:
            tensor = torch.frombuffer(memoryview(array), dtype=torch_dtype)
            return tensor.reshape(array.shape).clone()
        except Exception:
            pass

    # Last-resort path. It is slower, but this visualization samples only a
    # limited number of validation points and correctness is preferable to an
    # environment-specific crash.
    if np.issubdtype(array.dtype, np.bool_):
        return torch.tensor(array.tolist(), dtype=torch.bool)
    if np.issubdtype(array.dtype, np.integer):
        return torch.tensor(array.tolist(), dtype=torch.long)
    if np.issubdtype(array.dtype, np.floating):
        return torch.tensor(array.tolist(), dtype=torch.float32)
    raise TypeError(f"Unsupported NumPy dtype for tensor conversion: {array.dtype}")


def _compat_to_tensor(self, data):
    """Drop-in replacement matching Pointcept ToTensor output dtypes."""
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, str):
        return data
    if isinstance(data, (int, np.integer)):
        return torch.LongTensor([int(data)])
    if isinstance(data, (float, np.floating)):
        return torch.FloatTensor([float(data)])
    if isinstance(data, np.ndarray):
        tensor = _ndarray_to_tensor(data)
        if np.issubdtype(data.dtype, np.bool_):
            return tensor.bool()
        if np.issubdtype(data.dtype, np.integer):
            return tensor.long()
        if np.issubdtype(data.dtype, np.floating):
            return tensor.float()
        raise TypeError(f"Unsupported NumPy dtype: {data.dtype}")
    if isinstance(data, Mapping):
        return {key: self(value) for key, value in data.items()}
    if isinstance(data, Sequence):
        return [self(value) for value in data]
    raise TypeError(f"type {type(data)} cannot be converted to tensor")


def _probe_numpy_bridge() -> bool:
    """Return True when native torch.from_numpy works in this environment."""
    try:
        probe = np.zeros((2,), dtype=np.float32)
        torch.from_numpy(probe)
        return True
    except Exception as exc:
        print(
            "[WARN] Native torch.from_numpy is unavailable in this environment: "
            f"{type(exc).__name__}: {exc}"
        )
        return False


def main():
    native_ok = _probe_numpy_bridge()
    if native_ok:
        print("[INFO] Native torch.from_numpy probe succeeded.")
    else:
        print(
            "[INFO] Enabling visualization-only DLPack/buffer fallback for "
            "Pointcept ToTensor."
        )
        pointcept_transform.ToTensor.__call__ = _compat_to_tensor

    target = REPO_ROOT / "tools" / "visualize_point_text_feature_space.py"
    if not target.is_file():
        raise FileNotFoundError(target)

    # Preserve all CLI arguments supplied by the shell wrapper.
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
