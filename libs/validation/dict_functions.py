from __future__ import annotations
from functools import wraps
from typing import Callable, Mapping, TypeVar, Any, Dict, Iterable, Literal
import numpy as np

K = TypeVar("K")
V = TypeVar("V")
R = TypeVar("R")

KeysMode = Literal["strict", "intersection", "union"]


def dict_values_wrapper(f: Callable[[V, Any], R]) -> Callable[[Mapping[K, V], Any], Dict[K, R]]:
    """
    Decorator-like wrapper: the resulting function expects a dict/mapping first,
    then forwards extra args/kwargs to `f(value, *args, **kwargs)`.
    """
    @wraps(f)
    def wrapped(d: Mapping[K, V], *args: Any, **kwargs: Any) -> Dict[K, R]:
        return {k: f(v, *args, **kwargs) for k, v in d.items()}
    return wrapped


def dict_items_wrapper(f: Callable[[K, V, Any], R]) -> Callable[[Mapping[K, V], Any], Dict[K, R]]:
    @wraps(f)
    def wrapped(d: Mapping[K, V], *args: Any, **kwargs: Any) -> Dict[K, R]:
        return {k: f(k, v, *args, **kwargs) for k, v in d.items()}
    return wrapped


def _keys_intersection(dicts: Iterable[Mapping[K, Any]]) -> set[K]:
    it = iter(dicts)
    first = next(it, None)
    if first is None:
        return set()
    keys = set(first.keys())
    for d in it:
        keys &= set(d.keys())
    return keys


def _keys_union_ordered(dicts: Iterable[Mapping[K, Any]]) -> list[K]:
    seen: set[K] = set()
    out: list[K] = []
    for d in dicts:
        for k in d.keys():
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out


def _keys_ordered_from_first(first: Mapping[K, Any], allowed: set[K]) -> list[K]:
    # preserve first mapping order when possible
    return [k for k in first.keys() if k in allowed]


def dict_values_zip_wrapper(
    f: Callable[..., R],
    *,
    keys: KeysMode = "strict",
    fillvalue: Any = None,
) -> Callable[..., Dict[K, R]]:
    """
    Wrapper for calling f on values from multiple mappings aligned by key.

    Returned function signature:
        wrapped(d1, d2, ..., *args, **kwargs) -> dict[key] = f(v1, v2, ..., *args, **kwargs)

    keys:
      - "strict": require all dicts have exactly the same key set (else KeyError)
      - "intersection": use only keys present in all dicts (silently drops others)
      - "union": use union of keys; missing values replaced by `fillvalue`
    """
    @wraps(f)
    def wrapped(*dicts: Mapping[K, Any], **kwargs: Any) -> Dict[K, R]:
        if not dicts:
            return {}

        first = dicts[0]

        if keys == "strict":
            base = set(first.keys())
            for i, d in enumerate(dicts[1:], start=2):
                if set(d.keys()) != base:
                    missing = base - set(d.keys())
                    extra = set(d.keys()) - base
                    raise KeyError(
                        f"Key mismatch for mapping #{i}: "
                        f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
                    )
            key_list = list(first.keys())

            out: Dict[K, R] = {}
            for k in key_list:
                vals = [d[k] for d in dicts]  # all present
                out[k] = f(*vals, **kwargs)
            return out

        if keys == "intersection":
            kset = _keys_intersection(dicts)
            key_list = _keys_ordered_from_first(first, kset)

            out: Dict[K, R] = {}
            for k in key_list:
                vals = [d[k] for d in dicts]  # present in all by construction
                out[k] = f(*vals, **kwargs)
            return out

        if keys == "union":
            key_list = _keys_union_ordered(dicts)

            out: Dict[K, R] = {}
            for k in key_list:
                vals = [d.get(k, fillvalue) for d in dicts]
                out[k] = f(*vals, **kwargs)
            return out

        raise ValueError(f"Unknown keys mode: {keys!r}")

    return wrapped


def dict_items_zip_wrapper(
    f: Callable[..., R],
    *,
    keys: KeysMode = "strict",
    fillvalue: Any = None,
) -> Callable[..., Dict[K, R]]:
    """
    Like dict_values_zip_wrapper, but calls:
        f(key, v1, v2, ..., *args, **kwargs)
    """
    @wraps(f)
    def wrapped(*dicts: Mapping[K, Any], **kwargs: Any) -> Dict[K, R]:
        if not dicts:
            return {}

        values_wrapped = dict_values_zip_wrapper(
            lambda *vals, **kw: None,  # placeholder; we’ll expand below
            keys=keys,
            fillvalue=fillvalue,
        )

        # We reuse the key alignment logic by computing the aligned keys first.
        # (No need to be fancy; keep it readable.)
        first = dicts[0]
        if keys == "strict":
            base = set(first.keys())
            for i, d in enumerate(dicts[1:], start=2):
                if set(d.keys()) != base:
                    missing = base - set(d.keys())
                    extra = set(d.keys()) - base
                    raise KeyError(
                        f"Key mismatch for mapping #{i}: "
                        f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
                    )
            key_list = list(first.keys())
            get_vals = lambda k: [d[k] for d in dicts]

        elif keys == "intersection":
            kset = _keys_intersection(dicts)
            key_list = _keys_ordered_from_first(first, kset)
            get_vals = lambda k: [d[k] for d in dicts]

        elif keys == "union":
            key_list = _keys_union_ordered(dicts)
            get_vals = lambda k: [d.get(k, fillvalue) for d in dicts]

        else:
            raise ValueError(f"Unknown keys mode: {keys!r}")

        out: Dict[K, R] = {}
        for k in key_list:
            out[k] = f(k, *get_vals(k), **kwargs)
        return out

    return wrapped


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

def certain_component(x, idx):
    if idx in x:
        idx = np.where(x == idx)
        return idx[0]

@dict_items_wrapper
def certain_dict_component(k, v, idx, query, dim=2):

    comp = certain_component(query[k], idx)
    if comp is None:
        return None

    return ith_component(v, comp, dim=dim)
    
def timesn(x,n=100):
    return x*n
def nan_clip(x, max_value=100):
    return x if x < max_value else None

norm = dict_values_wrapper(np.linalg.norm)
none_norm = dict_values_wrapper(none_consistent_norm)
tn = dict_values_wrapper(timesn)
clip = dict_values_wrapper(nan_clip)
ith = dict_values_wrapper(ith_component)
# -------------------------
# Example with Metric
# -------------------------

# Suppose Metric instances are callable: metric(a,b) -> array
# difference = Difference()
# identity = IdentityStat()
# times = Times()

# Wrap metrics so they can be applied to dicts-of-arrays:
# - For arity=2: pass two dicts with same keys (strict by default)
# - For arity=1: pass one dict

# difference_on_dicts = dict_values_zip_wrapper(Difference(), keys="strict")
# out = difference_on_dicts(pred_dict, truth_dict)

# identity_on_dicts = dict_values_zip_wrapper(IdentityStat())
# out2 = identity_on_dicts(field_dict)

# times_on_dicts = dict_values_zip_wrapper(Times(), keys="intersection")
# out3 = times_on_dicts(a_dict, b_dict)  # computes only on common keys