from __future__ import annotations
from functools import wraps
from typing import Callable, Mapping, TypeVar, Any, Dict
from libs.validation.dict_functions import dict_items_wrapper
import numpy as np

def timesn(x,n=100):
    return x*n

def nan_clip(x, max_value=100):
    return x if x < max_value else None

def ith_component(x: Any, i: int, *, dim: int = 0, default: Any = None):
    """
    Return x indexed by `i` along a given dimension `dim`:
      - for arrays/tensors: x[..., i, ...] with i placed at `dim`
      - for 1D sequences: behaves like x[i] when dim is 0 or -1
    On TypeError/IndexError (not indexable / out of bounds), returns `default`.
    """
    try:
        # Determine number of dimensions for array/tensor-like objects
        if hasattr(x, "ndim"):
            ndim = int(x.ndim)  # NumPy, xarray, etc.
        elif hasattr(x, "dim") and callable(getattr(x, "dim")):
            ndim = int(x.dim())  # PyTorch tensor
        else:
            # Plain Python sequences: only meaningful "dim" is 0 (or -1 == 0 for 1D)
            if dim not in (0, -1):
                return default
            return x[i]

        # Normalize negative dim
        if dim < 0:
            dim += ndim
        if not (0 <= dim < ndim):
            return default

        # Build slicing tuple: [:, :, i, :, ...]
        sl = [slice(None)] * ndim
        sl[dim] = i
        return x[tuple(sl)]

    except (TypeError, IndexError, KeyError):
        return default

def none_consistent_norm(x, *args, **kwargs):
    if x is None:
        return None
    else:
        return np.linalg.norm(x, *args, **kwargs)

def certain_component(x, bid):
    if bid in x:
        idx = np.where(x == bid)
        return idx[0]

@dict_items_wrapper
def certain_dict_component(k, v, bid, query, dim=2):

    comp = certain_component(query[k], bid)
    if comp is None:
        return None

    return ith_component(v, comp, dim=dim)
    