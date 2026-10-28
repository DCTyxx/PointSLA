#!/usr/bin/env python3
"""Compatibility launcher for point/text feature-space visualization.

Some server environments expose NumPy arrays for which ``torch.from_numpy``
raises ``TypeError: expected np.ndarray (got numpy.ndarray)`` because the
installed PyTorch/NumPy bridge is ABI-incompatible or because arrays originate
from a different NumPy loading path.

This launcher patches conversions only inside this visualization process. It
uses DLPack first, then the Python buffer protocol / Python lists as fallbacks.
Training code and repository-wide transforms are not modified.
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


_ORIGINAL_TENSOR_NUMPY = torch.Tensor.numpy
_ORIGINAL_TENSOR_GETITEM = torch.Tensor.__getitem__


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

    if hasattr(array, "__dlpack__"):
        try:
            return torch.utils.dlpack.from_dlpack(array)
        except Exception:
            pass

    torch_dtype = _dtype_for_buffer(array.dtype)
    if torch_dtype is not None:
        try:
            tensor = torch.frombuffer(memoryview(array), dtype=torch_dtype)
            return tensor.reshape(array.shape).clone()
        except Exception:
            pass

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


def _compat_tensor_numpy(self, *args, **kwargs):
    """Torch -> NumPy fallback for the same ABI-incompatible environments."""
    try:
        return _ORIGINAL_TENSOR_NUMPY(self, *args, **kwargs)
    except Exception:
        tensor = self.detach().cpu().resolve_conj().resolve_neg()
        if hasattr(np, "from_dlpack"):
            try:
                # NumPy's DLPack view can be read-only; copy gives normal ndarray
                # semantics expected by scikit-learn/UMAP and np.savez.
                return np.from_dlpack(tensor).copy()
            except Exception:
                pass
        return np.asarray(tensor.tolist())


def _compat_index(index, device):
    """Convert NumPy indices before PyTorch tries to infer their dtype."""
    if isinstance(index, np.ndarray):
        converted = _ndarray_to_tensor(index)
        if np.issubdtype(index.dtype, np.bool_):
            converted = converted.bool()
        elif np.issubdtype(index.dtype, np.integer):
            converted = converted.long()
        return converted.to(device=device)
    if isinstance(index, np.integer):
        return int(index)
    if isinstance(index, tuple):
        return tuple(_compat_index(item, device) for item in index)
    return index


def _compat_tensor_getitem(self, index):
    """Allow Tensor[numpy_int_array] without using PyTorch's NumPy bridge."""
    return _ORIGINAL_TENSOR_GETITEM(self, _compat_index(index, self.device))


def _probe_numpy_bridge():
    try:
        probe = np.zeros((2,), dtype=np.float32)
        tensor = torch.from_numpy(probe)
        _ORIGINAL_TENSOR_NUMPY(tensor)
        return True, None
    except Exception as exc:
        return False, exc


def main():
    native_ok, native_exc = _probe_numpy_bridge()
    if native_ok:
        print(
            "[INFO] Native NumPy/Torch bridge probe succeeded; "
            "using the visualization-local compatibility converter anyway "
            "to avoid dataset-specific bridge failures."
        )
    else:
        print(
            "[WARN] Native NumPy/Torch bridge probe failed: "
            f"{type(native_exc).__name__}: {native_exc}"
        )

    # Patches are process-local. Existing ToTensor objects dispatch __call__
    # through their class, so already-built Compose pipelines use this method.
    pointcept_transform.ToTensor.__call__ = _compat_to_tensor
    torch.Tensor.numpy = _compat_tensor_numpy
    torch.Tensor.__getitem__ = _compat_tensor_getitem
    print(
        "[INFO] Enabled visualization-only DLPack/buffer compatibility for "
        "NumPy <-> Torch conversion and NumPy-based Tensor indexing."
    )

    target = REPO_ROOT / "tools" / "visualize_point_text_feature_space.py"
    if not target.is_file():
        raise FileNotFoundError(target)

    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
