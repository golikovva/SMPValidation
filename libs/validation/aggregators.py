import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Sequence, Iterable
import warnings
import torch
from libs.validation.inv_dist_interp import InvDistTree


class Aggregator(ABC):
    """
    Abstract base class for aggregators that collect statistics from pointwise fields.
    """
    def __init__(self, space_arity=2):
        super().__init__()
        self.space_arity = space_arity


    @abstractmethod
    def init_accumulator(self, shape: Tuple[int, ...]) -> Any:
        """
        Initialize and return an accumulator structure appropriate for this aggregator.
        :param shape: The shape of incoming fields (e.g., spatial grid shape).
        """
        pass

    @abstractmethod
    def accumulate(self, acc: Any, field: np.ndarray, date: Any) -> None:
        """
        Update the accumulator with a new field at a given date.
        :param acc: The accumulator returned by init_accumulator.
        :param field: The pointwise data array to aggregate.
        :param date: The corresponding date or time identifier.
        """
        pass

    @staticmethod
    @abstractmethod
    def finalize(acc: Any) -> Any:
        """
        Compute and return the final aggregated result from the accumulator.
        """
        pass


class GlobalTemporalAggregator(Aggregator):
    """
    Computes the global spatial mean at each date and stores a time series.
    """

    def init_accumulator(self, shape: Tuple[int, ...]) -> Dict[Any, float]:
        # Use a dict date -> mean value
        return {}

    def accumulate(self, acc: Dict[Any, float], field: np.ndarray, date: Any) -> None:
        if field.ndim < self.space_arity:
            raise ValueError(
                f"field.ndim={field.ndim} is smaller than space_arity={self.space_arity}. "
                f"Cannot reduce last {self.space_arity} dims."
            )
        space_axes = tuple(range(field.ndim - self.space_arity, field.ndim))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            mean_over_space = np.nanmean(field, axis=space_axes)
        acc[date] = {'sum': np.nansum(field, axis=space_axes), 'count': np.sum(~np.isnan(field), axis=space_axes)}
        # acc[date] = mean_over_space
    
    @staticmethod
    def finalize(acc: Dict[Any, np.ndarray]) -> Dict[Any, np.ndarray]:
        return {date: (v['sum'] / v['count']) if v['count'] > 0 else np.nan for date, v in acc.items()}
        # return acc


class GlobalIntegratedTemporalAggregator(GlobalTemporalAggregator):
    """
    Computes the global spatial sum at each date and stores a time series.
    """
    def __init__(self, area_map=None):
        super().__init__()
        self.area_map = area_map

    def accumulate(self, acc, field, date):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            if self.area_map is not None:
                field = field*self.area_map
            mean_over_space = np.nansum(field, axis=(-2, -1))
        acc[date] = mean_over_space


class RegionalTemporalAggregator(Aggregator):
    """
    Accumulate per‐region, per‐date means of a C×H×W field.

    Parameters
    ----------
    region_mask : np.ndarray[int], shape (H, W)
        0 = skip, >0 = region ID.
    region_ids : Sequence[int], optional
        Which labels to track; default = all unique >0 in mask.
    """
    def __init__(self,
                 region_mask: np.ndarray,
                 region_ids=None):
        self.region_mask = region_mask
        # derive region IDs if not given
        if region_ids is None:
            self.region_ids = np.unique(region_mask)
            self.region_ids = [int(r) for r in self.region_ids if r != 0]
        else:
            self.region_ids = list(region_ids)
        self.max_label = int(region_mask.max())

    def init_accumulator(self, shape):
        """
        Returns a dict:
          region_id -> { date1: np.ndarray(C,), date2: np.ndarray(C,), ... }
        """
        return {rid: {} for rid in self.region_ids}

    def accumulate(self, acc: Dict[int, Dict[Any, np.ndarray]],
                   field: np.ndarray,
                   date: Any) -> None:
        """
        field must have last dims (C, H, W).
        We flatten H×W, then for each channel c do a bincount over region_mask.
        """
        if field.ndim == 2:
            field = field[None]  # 2D field, assume (H, W)
        if field.ndim < 3:
            raise ValueError("Field must have at least 3 dims (C, H, W) at the end")
        C, H, W = field.shape[-3], field.shape[-2], field.shape[-1]
        if (H, W) != self.region_mask.shape:
            raise ValueError(f"Field spatial dims {(H,W)} != mask shape {self.region_mask.shape}")

        # Flatten
        flat_mask = self.region_mask.ravel()  # (H*W,)
        flat_data = field.reshape(C, -1)      # (C, H*W)

        # Prepare sums & counts arrays
        sums = np.zeros((C, self.max_label+1), dtype=float)
        counts = np.zeros((C, self.max_label+1), dtype=int)

        for c in range(C):
            vals = flat_data[c]
            # Sum (NaNs→0)
            sums[c] = np.bincount(
                flat_mask,
                weights=np.nan_to_num(vals, nan=0.0),
                minlength=self.max_label+1
            )
            # Count valid
            valid_idx = ~np.isnan(vals)
            counts[c] = np.bincount(
                flat_mask[valid_idx],
                minlength=self.max_label+1
            )

        # Compute & store per‐region mean vectors
        for rid in self.region_ids:
            # avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                mean_vec = sums[:, rid] / counts[:, rid]
            mean_vec[counts[:, rid] == 0] = np.nan
            acc[rid][date] = {'sum': sums[:, rid], 'count': counts[:, rid]}
            # acc[rid][date] = mean_vec

    @staticmethod
    def finalize(acc: Dict[Any, np.ndarray]) -> Dict[Any, np.ndarray]:
        return {rid: {date: (v['sum'] / v['count']) if v['count'] > 0 else np.nan for date, v in date_dict.items()} for rid, date_dict in acc.items()}
        # return acc


class RegionalIntegratedTemporalAggregator(RegionalTemporalAggregator):
    """
    Computes the regional spatial sum at each date and stores a time series.
    """
    def __init__(self,
                 region_mask: np.ndarray,
                 region_ids=None,
                 area_map=None):
        super().__init__(region_mask, region_ids)
        self.area_map = area_map

    def accumulate(self, acc: Dict[int, Dict[Any, np.ndarray]],
                    field: np.ndarray,
                    date: Any) -> None:
            """
            field must have last dims (C, H, W).
            We flatten H×W, then for each channel c do a bincount over region_mask.
            """
            if field.ndim == 2:
                field = field[None]  # 2D field, assume (H, W)
            if field.ndim < 3:
                raise ValueError("Field must have at least 3 dims (C, H, W) at the end")
            C, H, W = field.shape[-3], field.shape[-2], field.shape[-1]
            if (H, W) != self.region_mask.shape:
                raise ValueError(f"Field spatial dims {(H,W)} != mask shape {self.region_mask.shape}")
            
            if self.area_map is not None:
                
                field = field*self.area_map
            # Flatten
            flat_mask = self.region_mask.ravel()  # (H*W,)
            flat_data = field.reshape(C, -1)      # (C, H*W)

            # Prepare sums & counts arrays
            sums = np.zeros((C, self.max_label+1), dtype=float)

            for c in range(C):
                vals = flat_data[c]
                # Sum (NaNs→0)
                sums[c] = np.bincount(
                    flat_mask,
                    weights=np.nan_to_num(vals, nan=0.0),
                    minlength=self.max_label+1
                )

            # Compute & store per‐region mean vectors
            for rid in self.region_ids:
                # avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    sum_vec = sums[:, rid] 
                acc[rid][date] = sum_vec


class SpatialAggregator(Aggregator):
    """
    Computes the mean field over time for each grid cell (i.e., time-averaged spatial map).
    """

    def init_accumulator(self, shape: Tuple[int, ...]) -> Dict[str, np.ndarray]:
        return {
            'sum': np.zeros(shape, dtype=float),
            'count': np.zeros(shape, dtype=int)
        }

    def accumulate(self, acc: Dict[str, np.ndarray], field: np.ndarray, date: Any) -> None:
        # Sum and count non-nan values
        sum_axis = tuple(range(0, field.ndim-2))
        acc['sum'] += np.nan_to_num(field, nan=0.0)
        acc['count'] += (~np.isnan(field)).astype(int)

    @staticmethod
    def finalize(acc: Dict[str, np.ndarray]) -> np.ndarray:
        mean_field = acc['sum'] / np.where(acc['count'] == 0, np.nan, acc['count'])
        mean_field[acc['count'] == 0] = np.nan
        return mean_field
    
class AverageAggregator(SpatialAggregator):
    """
    Computes the mean of all accumulated fields, ignoring NaNs.
    """
    def init_accumulator(self, shape: Tuple[int, ...]) -> Dict[str, np.ndarray]:
        return {
            'sum': np.zeros(1, dtype=float),
            'count': np.zeros(1, dtype=int)
        }
    def accumulate(self, acc: Dict[str, np.ndarray], field: np.ndarray, date: Any) -> None:
        # Sum and count non-nan values
        sum_axis = tuple(range(0, field.ndim-2))
        acc['sum'] += np.nan_to_num(field, nan=0.0).sum()
        acc['count'] += (~np.isnan(field)).astype(int).sum()

    @staticmethod
    def finalize(acc: Dict[str, np.ndarray]) -> np.ndarray:
        return np.nansum(acc['sum']) / np.nansum(acc['count'])


class SeasonalSpatialAggregator(Aggregator):
    """
    Aggregate 2D fields into user‑defined “seasons” (arbitrary sets of months).

    Parameters
    ----------
    season_months : Dict[str, Sequence[int]]
        Mapping from season name (e.g. 'FMA', 'DJF', 'Jan', 'All') to the
        list/tuple of month numbers (1–12) that belong to that season.
        E.g.:
            {
              'FMA': (2,3,4),
              'MJJ': (5,6,7),
              'ASO': (8,9,10),
              'NDJ': (11,12,1),
            }
    """
    DEFAULT_SEASONS = {
        'FMA': (2, 3, 4),
        'MJJ': (5, 6, 7),
        'ASO': (8, 9, 10),
        'NDJ': (11, 12, 1),
    }

    def __init__(self, season_months: Dict[str, Sequence[int]] = None):
        # Use default if not provided
        self.season_months = (
            {**self.DEFAULT_SEASONS}
            if season_months is None
            else {name: tuple(months) for name, months in season_months.items()}
        )

        # Validate months and build reverse lookup
        self._month_to_seasons: Dict[int, Tuple[str, ...]] = {}
        for name, months in self.season_months.items():
            for m in months:
                if not (1 <= m <= 12):
                    raise ValueError(f"Season '{name}' has invalid month {m}")
                self._month_to_seasons.setdefault(m, []).append(name)

    def init_accumulator(self, shape: Tuple[int, ...]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        :param shape: spatial shape of each incoming 2D field (H, W).
        Returns a dict:
            season_name -> {'sum': ndarray(H,W), 'count': ndarray(H,W)}
        """
        acc: Dict[str, Dict[str, np.ndarray]] = {}
        for season in self.season_months:
            acc[season] = {
                'sum':   np.zeros((shape), dtype=float),
                'count': np.zeros((shape), dtype=int),
            }
        return acc

    def accumulate(self,
                   acc: Dict[str, Dict[str, np.ndarray]],
                   field: np.ndarray,
                   date: Any) -> None:
        """
        Add this 2D field into all seasons that contain date.month.

        :param field: 2D array (H, W) of data or error.
        :param date: datetime.date (or anything with `.month`).
        """

        month = date.month
        seasons = self._month_to_seasons.get(month)
        if not seasons:
            raise ValueError(f"No season defined for month={month}")

        # Prepare sum/count update
        field_sum   = np.nan_to_num(field, nan=0.0)
        field_count = (~np.isnan(field)).astype(int)

        for season in seasons:
            acc_season = acc[season]
            acc_season['sum']   += field_sum
            acc_season['count'] += field_count
            
    @staticmethod
    def finalize(acc: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Return a dict:
            season_name -> mean_map (2D ndarray)
        """
        result: Dict[str, np.ndarray] = {}
        for season, data in acc.items():
            s = data['sum']
            c = data['count']
            with np.errstate(divide='ignore', invalid='ignore'):
                m = s / c
            m[c == 0] = np.nan
            result[season] = m
        return result
    

class CoordinateTemporalAggregator(Aggregator):
    """
    Для каждого набора координат из coords_dict строит интерполяционное дерево,
    а затем накапливает среднее по этим точкам для каждого канала.
    
    Parameters
    ----------
    grid : object
        Объект с атрибутами `.lat` и `.lon` — двумерными массивами формы (H, W).
    coords_dict : Dict[str, np.ndarray]
        Словарь "имя набора" → массив точек shape (N_points, 2) в тех же единицах,
        что и grid.lat/lon (например, градусы).
    leaf_size, n_near, sigma_squared, distance_metric, inv_dist_mode, device
        Параметры проксейдерева для InvDistTree (см. класс InvDistTree).
    """

    def __init__(self,
                 grid: Any,
                 coords_dict: Dict[str, np.ndarray],
                 leaf_size: int = 10,
                 n_near: int = 6,
                 sigma_squared: float = None,
                 distance_metric: str = 'euclidean',
                 inv_dist_mode: str = 'gaussian',
                 device: str = 'cpu'):
        # Сохраним словарь имен
        self.names = list(coords_dict.keys())
        # Подготовим массив исходных точек X: (H*W, 2)
        lat = grid.lat    # shape (H, W)
        lon = grid.lon    # shape (H, W)
        H, W = lat.shape
        X = np.stack([lat.ravel(), lon.ravel()], axis=1)  # (H*W, 2)

        # Для каждого набора целевых точек Q строим InvDistTree
        self.trees: Dict[str, InvDistTree] = {}
        for name, Q in coords_dict.items():
            # Q: np.ndarray shape (N_points, 2)
            tree = InvDistTree(
                x=X,
                q=Q,
                leaf_size=leaf_size,
                n_near=n_near,
                sigma_squared=sigma_squared,
                distance_metric=distance_metric,
                inv_dist_mode=inv_dist_mode,
                device=device,
                has_nans=True,
            )
            self.trees[name] = tree

    def init_accumulator(self, shape: Tuple[int, ...]) -> Dict[str, Dict[Any, np.ndarray]]:
        """
        Игнорируем `shape`, возвращаем словарь:
          name -> {}  
        где под каждым именем накапливаем date -> np.ndarray(C,)
        """
        return {name: {} for name in self.names}

    def accumulate(self,
                   acc: Dict[str, Dict[Any, np.ndarray]],
                   field: np.ndarray,
                   date: Any) -> None:
        """
        Для каждого набора точек:
          1) интерполируем field (C×H×W) на tree
          2) усредняем по последней размерности (точки)
          3) сохраняем в acc[name][date] = вектор length C
        """
        # Проверяем форму: должно быть как минимум 3D: (..., C,H,W), но мы ожидаем ровно C×H×W
        arr = np.asarray(field)
        if arr.ndim != 3:
            raise ValueError(f"Field must be 3-dim (C,H,W), got {arr.shape}")
        C, H, W = arr.shape

        for name, tree in self.trees.items():
            # Переводим в torch.Tensor на тот же device, что и веса дерева
            # dtype=float32
            z = torch.as_tensor(arr, dtype=tree.weights.dtype, device=tree.weights.device)
            print(z.shape)
            # Интерполируем: результат shape (C, N_points)
            vals = tree(z.flatten(-2, -1))
            print(vals.shape)
            # Усредняем по точкам → shape (C,)
            mean_vec = vals.cpu().numpy()
            # Сохраняем
            acc[name][date] = mean_vec

    @staticmethod
    def finalize(acc: Dict[str, Dict[Any, np.ndarray]]) -> Dict[str, Dict[Any, np.ndarray]]:
        # Просто возвращаем накопленную структуру
        return acc


class RawFieldAggregator(Aggregator):
    """
    Passes through the raw field values, collecting them as a time series of full fields.
    """

    def init_accumulator(self, shape: Tuple[int, ...]) -> Dict[Any, np.ndarray]:
        return {}

    def accumulate(self, acc: Dict[Any, np.ndarray], field: np.ndarray, date: Any) -> None:
        acc[date] = field.copy()

    @staticmethod
    def finalize(acc: Dict[Any, np.ndarray]) -> Dict[Any, np.ndarray]:
        return acc
    

class BinnedByConditionAggregator(Aggregator):
    """
    Агрегирует field по бинам condition_dataset[date].

    Например:
      field = MAE(model, obs)
      condition = cloud_fraction[date]

    На выходе:
      mean error as a function of cloud fraction
    """

    def __init__(self, condition_dataset, bins, right: bool = False, space_arity: int = 2):
        super().__init__(space_arity=space_arity)
        self.condition_dataset = condition_dataset
        self.bins = np.asarray(bins, dtype=float)
        self.right = right

        if self.bins.ndim != 1 or len(self.bins) < 2:
            raise ValueError("bins must be a 1D array with at least 2 edges")

    def init_accumulator(self, shape: Tuple[int, ...]) -> Dict[str, np.ndarray]:
        nbins = len(self.bins) - 1
        return {
            "sum": np.zeros(nbins, dtype=float),
            "sum_sq": np.zeros(nbins, dtype=float),
            "count": np.zeros(nbins, dtype=int),
        }

    def accumulate(self, acc: Dict[str, np.ndarray], field: np.ndarray, date: Any) -> None:
        try:
            cond = self.condition_dataset[date]
        except Exception:
            return

        if cond is None or field is None:
            return

        field = np.asarray(field, dtype=float)
        cond = np.asarray(cond, dtype=float)

        value = field.ravel()
        cond_val = cond.ravel()

        if value.shape != cond_val.shape:
            raise ValueError(
                f"Shape mismatch in BinnedByConditionAggregator for date={date}: "
                f"field.shape={field.shape}, cond.shape={cond.shape}"
            )

        valid = np.isfinite(value) & np.isfinite(cond_val)
        if not np.any(valid):
            return

        value = value[valid]
        cond_val = cond_val[valid]

        idx = np.digitize(cond_val, self.bins, right=self.right) - 1
        good = (idx >= 0) & (idx < len(self.bins) - 1)

        if not np.any(good):
            return

        value = value[good]
        idx = idx[good]

        np.add.at(acc["sum"], idx, value)
        np.add.at(acc["sum_sq"], idx, value**2)
        np.add.at(acc["count"], idx, 1)

    @staticmethod
    def finalize(acc: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        mean = np.full_like(acc["sum"], np.nan, dtype=float)
        std = np.full_like(acc["sum"], np.nan, dtype=float)

        mask = acc["count"] > 0
        mean[mask] = acc["sum"][mask] / acc["count"][mask]

        var = np.full_like(acc["sum"], np.nan, dtype=float)
        var[mask] = acc["sum_sq"][mask] / acc["count"][mask] - mean[mask]**2
        var[mask] = np.maximum(var[mask], 0.0)
        std[mask] = np.sqrt(var[mask])

        return {
            "mean": mean,
            "std": std,
            "count": acc["count"].copy(),
        }
    

class BinnedByConditionPerDateAggregator(BinnedByConditionAggregator):
    """
    То же самое, что BinnedByConditionAggregator, но считает статистику
    отдельно для каждой даты.

    Результат finalize:
    {
        "bins": ...,
        "by_date": {
            date1: {"mean": ..., "std": ..., "count": ...},
            date2: {"mean": ..., "std": ..., "count": ...},
            ...
        }
    }
    """

    def init_accumulator(self, shape: Tuple[int, ...]) -> Dict[str, Any]:
        return {
            "bins": self.bins.copy(),
            "by_date": {}
        }

    def accumulate(self, acc: Dict[str, Any], field: np.ndarray, date: Any) -> None:
        if date not in acc["by_date"]:
            acc["by_date"][date] = super().init_accumulator(field.shape)

        super().accumulate(acc["by_date"][date], field, date)

    @staticmethod
    def finalize(acc: Dict[str, Any]) -> Dict[str, Any]:
        by_date_final = {}
        for date, date_acc in acc["by_date"].items():
            by_date_final[date] = BinnedByConditionAggregator.finalize(date_acc)

        return {
            "bins": acc["bins"].copy(),
            "by_date": by_date_final,
        }
    

def get_field_meta(field, key=None, default=None):
    meta = getattr(field, "meta", None)
    if meta is None:
        return default if key is not None else {}
    if key is None:
        return meta
    return meta.get(key, default)


class MetricField(np.ndarray):
    """
    ndarray + metadata
    """
    def __new__(cls, input_array, **meta):
        obj = np.asarray(input_array).view(cls)
        obj.meta = dict(meta)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.meta = getattr(obj, "meta", {})

    def __reduce__(self):
        """Keep metadata alongside NumPy's array state when saving results."""
        reconstruct, args, state = super().__reduce__()
        return reconstruct, args, state + (getattr(self, "meta", {}),)

    def __setstate__(self, state):
        # Older result files contain only NumPy's ordinary ndarray state.
        self.meta = state[-1] if len(state) == 6 else {}
        array_state = state[:-1] if len(state) == 6 else state
        super().__setstate__(array_state)


class NestedLeadTimeAggregator(Aggregator):
    """
    Split field by lead_time from field.meta[lead_key] and delegate each subset
    to nested aggregators.

    Example
    -------
    field.shape = (T, R, C, H, W)
    field.meta["lead_h"].shape = (T, R)

    For each unique lead value L:
        mask = (lead_h == L)
        sub_field = field[mask]   # shape (N_selected, C, H, W)

    Then each nested aggregator receives sub_field.

    Result structure:
        {
            lead_1: {
                "SpatialAggregator": ...,
                "AverageAggregator": ...,
            },
            lead_2: {
                ...
            }
        }

    Notes
    -----
    1. This aggregator collapses all lead-indexed prefix axes selected by mask
       into one leading batch axis via boolean indexing.
    2. Nested aggregators must tolerate an additional leading batch dimension.
       Most of your current aggregators do.
    """

    def __init__(
        self,
        aggregators: Iterable,
        lead_key: str = "lead_h",
        mask_key: str = "avail_mask",
        ignore_negative_leads: bool = True,
    ):
        normalized = []
        for i, item in enumerate(aggregators):
            if isinstance(item, tuple):
                name, agg = item
            else:
                agg = item
                name = f"{agg.__class__.__name__}__{i}"
            normalized.append((name, agg))

        self.nested_aggregators: List[Tuple[str, Aggregator]] = normalized
        self.lead_key = lead_key
        self.mask_key = mask_key
        self.ignore_negative_leads = ignore_negative_leads

    def init_accumulator(self, shape: Tuple[int, ...]) -> Dict[str, Any]:
        return {
            "by_lead": {}
        }

    def _make_subfield(self, field: np.ndarray, mask: np.ndarray, lead_value: int):
        arr = np.asarray(field)
        sub_arr = arr[mask]  # -> (N_selected, *suffix)

        # Try to preserve relevant metadata in filtered form
        meta = {}
        src_meta = get_field_meta(field, None, {})

        # lead_h is now constant on selected subset
        meta[self.lead_key] = np.full((sub_arr.shape[0],), int(lead_value), dtype=np.int32)

        # carry selected avail_mask if needed (now always True for selected entries)
        if self.mask_key in src_meta:
            meta[self.mask_key] = np.ones((sub_arr.shape[0],), dtype=bool)

        return MetricField(sub_arr, **meta)

    def accumulate(self, acc: Dict[str, Any], field: np.ndarray, date: Any) -> None:
        lead_h = get_field_meta(field, self.lead_key, None)
        if lead_h is None:
            lead_h = 0
            # raise ValueError(
            #     f"{self.__class__.__name__} requires field.meta['{self.lead_key}']"
            # )

        lead_h = np.asarray(lead_h)
        arr = np.asarray(field)

        if arr.shape[:lead_h.ndim] != lead_h.shape:
            raise ValueError(
                f"{self.__class__.__name__}: lead_h shape {lead_h.shape} must match "
                f"prefix of field.shape {arr.shape}"
            )

        avail_mask = get_field_meta(field, self.mask_key, None)
        if avail_mask is not None:
            avail_mask = np.asarray(avail_mask, dtype=bool)
            if avail_mask.shape != lead_h.shape:
                raise ValueError(
                    f"{self.__class__.__name__}: avail_mask shape {avail_mask.shape} "
                    f"must match lead_h shape {lead_h.shape}"
                )

        unique_leads = np.unique(lead_h)
        if self.ignore_negative_leads:
            unique_leads = unique_leads[unique_leads >= 0]

        for lead in unique_leads:
            mask = (lead_h == lead)
            if avail_mask is not None:
                mask = mask & avail_mask

            if not np.any(mask):
                continue

            sub_field = self._make_subfield(field, mask, int(lead))

            lead_key = int(lead)
            if lead_key not in acc["by_lead"]:
                acc["by_lead"][lead_key] = {}

            lead_bucket = acc["by_lead"][lead_key]

            for agg_name, agg in self.nested_aggregators:
                if agg_name not in lead_bucket:
                    lead_bucket[agg_name] = agg.init_accumulator(sub_field.shape)
                agg.accumulate(lead_bucket[agg_name], sub_field, date)

    def finalize(self, acc: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        result = {}
        for lead in sorted(acc["by_lead"].keys()):
            result[lead] = {}
            for agg_name, agg in self.nested_aggregators:
                if agg_name in acc["by_lead"][lead]:
                    result[lead][agg_name] = agg.finalize(acc["by_lead"][lead][agg_name])
        return result