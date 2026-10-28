#!/usr/bin/env python3
"""Compatibility launcher for point/text feature-space visualization.

Some server environments expose NumPy arrays for which ``torch.from_numpy``
raises ``TypeError: expected np.ndarray (got numpy.ndarray)`` because the
installed PyTorch/NumPy bridge is ABI-incompatible or because arrays originate
from a different NumPy loading path.

This launcher patches conversions only inside this visualization process. It
uses DLPack first, then the Python buffer protocol / Python lists as fallbacks.
It also bypasses NumPy's ``savez`` array-function dispatcher, which can fail in
mixed/ABI-incompatible NumPy environments even after feature extraction has
completed successfully. Training code and repository-wide transforms are not
modified.
"""

from __future__ import annotations

import io
import runpy
import sys
import zipfile
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
    """Convert Torch -> NumPy without calling PyTorch's NumPy C-API bridge."""
    if args or kwargs:
        raise TypeError(
            "Visualization compatibility Tensor.numpy() does not accept arguments"
        )

    tensor = self.detach().cpu().resolve_conj().resolve_neg().contiguous()
    if hasattr(np, "from_dlpack"):
        try:
            # NumPy's DLPack view can be read-only. Copy gives ordinary ndarray
            # semantics expected by scikit-learn, UMAP, plotting, and saving.
            return np.from_dlpack(tensor).copy()
        except Exception:
            pass

    # Slow last resort, used only when DLPack is unavailable for this dtype.
    values = tensor.tolist()
    if tensor.dtype == torch.bool:
        return np.array(values, dtype=np.bool_)
    if tensor.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        dtype_map = {
            torch.uint8: np.uint8,
            torch.int8: np.int8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
        }
        return np.array(values, dtype=dtype_map[tensor.dtype])
    if tensor.dtype in (torch.float16, torch.float32, torch.float64):
        dtype_map = {
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.float64: np.float64,
        }
        return np.array(values, dtype=dtype_map[tensor.dtype])
    return np.array(values)


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


def _compat_savez(file, *args, **kwargs):
    """Write a normal NPZ archive without NumPy's __array_function__ dispatcher."""
    arrays = {}
    for idx, value in enumerate(args):
        arrays[f"arr_{idx}"] = value
    for key, value in kwargs.items():
        if key in arrays:
            raise ValueError(f"Duplicate NPZ key: {key}")
        arrays[str(key)] = value

    path = file
    if isinstance(path, (str, Path)):
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"

    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for key, value in arrays.items():
            # The analysis script passes ndarrays here. Avoid np.asarray on an
            # ndarray because that may invoke the same broken dispatch stack.
            array = value if isinstance(value, np.ndarray) else np.array(value)
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=True)
            archive.writestr(f"{key}.npy", buffer.getvalue())


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

    # All patches are process-local. Existing ToTensor objects dispatch __call__
    # through their class, so already-built Compose pipelines use this method.
    pointcept_transform.ToTensor.__call__ = _compat_to_tensor
    torch.Tensor.numpy = _compat_tensor_numpy
    torch.Tensor.__getitem__ = _compat_tensor_getitem
    np.savez = _compat_savez
    print(
        "[INFO] Enabled visualization-only compatibility for NumPy <-> Torch, "
        "NumPy-based Tensor indexing, and NPZ saving."
    )

    target = REPO_ROOT / "tools" / "visualize_point_text_feature_space.py"
    if not target.is_file():
        raise FileNotFoundError(target)

    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
