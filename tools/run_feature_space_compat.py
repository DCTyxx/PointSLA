#!/usr/bin/env python3
"""Compatibility launcher for point/text feature-space visualization.

Some server environments expose NumPy arrays for which ``torch.from_numpy``
raises ``TypeError: expected np.ndarray (got numpy.ndarray)`` because the
installed PyTorch/NumPy bridge is ABI-incompatible or because arrays originate
from a different NumPy loading path.

This launcher patches only Pointcept's ``ToTensor`` class in this process.  It
prefers NumPy's DLPack interface, falls back to the Python buffer protocol, and
finally to a list conversion.  Training code and repository-wide transforms are
not modified.
"""

from __future__ import annotations

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

    # Preferred path: NumPy >= 1.22 implements DLPack. This bypasses PyTorch's
    # NumPy C-API bridge and is efficient for normal numeric arrays.
    if hasattr(array, "__dlpack__"):
        try:
            return torch.utils.dlpack.from_dlpack(array)
        except Exception:
            pass

    # Second path: Python buffer protocol. Clone so the tensor owns stable
    # storage independent of the temporary contiguous NumPy array.
    torch_dtype = _dtype_for_buffer(array.dtype)
    if torch_dtype is not None:
        try:
            tensor = torch.frombuffer(memoryview(array), dtype=torch_dtype)
            return tensor.reshape(array.shape).clone()
        except Exception:
            pass

    # Last resort. Slower, but robust and only used in this diagnostic script.
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


def _probe_numpy_bridge():
    try:
        probe = np.zeros((2,), dtype=np.float32)
        torch.from_numpy(probe)
        return True, None
    except Exception as exc:
        return False, exc


def main():
    native_ok, native_exc = _probe_numpy_bridge()
    if native_ok:
        print(
            "[INFO] Native torch.from_numpy probe succeeded; "
            "using the visualization-local compatibility converter anyway "
            "to avoid dataset-specific NumPy bridge failures."
        )
    else:
        print(
            "[WARN] Native torch.from_numpy probe failed: "
            f"{type(native_exc).__name__}: {native_exc}"
        )

    # Always patch in this process only. Existing ToTensor objects dispatch
    # __call__ through their class, so already-built Compose pipelines use this
    # replacement as well.
    pointcept_transform.ToTensor.__call__ = _compat_to_tensor
    print(
        "[INFO] Enabled visualization-only DLPack/buffer fallback for "
        "Pointcept ToTensor."
    )

    target = REPO_ROOT / "tools" / "visualize_point_text_feature_space.py"
    if not target.is_file():
        raise FileNotFoundError(target)

    # Preserve all CLI arguments supplied by the shell wrapper.
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
