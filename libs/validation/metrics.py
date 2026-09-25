from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
from libs.validation.models.partial_conv_2d import local_std
import numpy as np
import torch
from pyproj import Geod


class Metric(ABC):
    """
    Abstract base class for metrics computing pointwise fields.
    Each metric must specify its name and arity (number of inputs).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of the metric."""
        pass

    @property
    @abstractmethod
    def arity(self) -> int:
        """Number of input fields required (e.g., 1 or 2)."""
        pass

    @abstractmethod
    def compute(self, *fields: np.ndarray) -> np.ndarray:
        """
        Compute the pointwise result array given the required number of input fields.
        :param fields: Tuple of numpy arrays of length self.arity
        :return: numpy array of computed errors/statistics
        """
        pass

    def __call__(self, *fields: np.ndarray, **kwargs) -> np.ndarray:
        """
        Call the compute method with the provided fields.
        :param fields: Tuple of numpy arrays of length self.arity
        :return: numpy array of computed errors/statistics
        """
        return self.compute(*fields, **kwargs)

class MSE(Metric):
    name = 'mse'
    arity = 2
    def compute(self, a, b):
        return (a - b)**2


class MAE(Metric):
    name = 'mae'
    arity = 2
    def compute(self, a, b):
        return np.abs(a - b)

class SkillScore(Metric):
    name = 'skill_score'
    arity = 2
    def compute(self, model_err, ref_err):
        with np.errstate(divide='ignore', invalid='ignore'):
            return ((ref_err - model_err) / ref_err)

class ReversedSkillScore(Metric):
    name = 'reversed_skill_score'
    arity = 3
    def compute(self, model_score, ref_score, perfect_score=1.0):
        with np.errstate(divide='ignore', invalid='ignore'):
            return (model_score - ref_score) / (perfect_score - ref_score)


class Difference(Metric):
    name = 'difference'
    arity = 2
    def compute(self, a, b):
        return a - b
    

class IdentityStat(Metric):
    name = 'identity'
    arity = 1
    def compute(self, a):
        return a 

class Times(Metric):
    name = 'times'
    arity = 2
    def compute(self, a, b):
        return a*b

class Net(Metric):
    name = 'net'
    arity = 2
    def compute(self, a, b):
        return a*(1 - b)

class StatTransformed(Metric):
    """
    Applies a transformation to the input array.
    """
    name = 'transformed'
    @property
    def arity(self):
        return self._arity

    def __init__(self, transform_fn, arity=1, name=None):
        self.transform_fn = transform_fn
        self._arity = arity
        if name is not None:
            self.name = name

    def compute(self, *a):
        return _unpack_tuple(tuple(self.transform_fn(item) for item in a))
    

class StatSquared(StatTransformed):  # usefull to calculate variance
    name = 'sqared'
    def __init__(self):
        self.transform_fn = lambda x: x**2

    
# class ApplyToAll(Metric):
#     """
#     Applies a given metric to each input field separately.
#     """
#     arity = None  # can be set to any value, will be determined by the wrapped metric
#     def __init__(self, metric: Metric, arity=None):
#         self.metric = metric
#         self._arity = arity

#     @property
#     def name(self):
#         return f"apply_to_all({self.metric.name})"

#     @property
#     def arity(self):
#         return self._arity or self.metric.arity

#     def compute(self, *fields):
#         return tuple(self.metric.compute(f) for f in fields)
    

class IdentityOver(Metric):
    """
    Returns a where b is not nan.
    """
    name = 'identity_over'
    arity = 2
    def compute(self, a, b):
        return np.where(np.isnan(b), np.nan, a)

class AngleError(Metric):
    """
    Computes the angular error between two vectors using the cosine of the angle between them.
    """
    name = 'angle_error'
    arity = 2
    def __init__(self, var_axis=-3, ):
        self.var_axis = var_axis
        self.transform_fn = lambda x: x**2

    def compute(self, first: np.ndarray, second: np.ndarray, sensitivity=0.1) -> np.ndarray:
        # Compute the L2 norm of the first and second arrays
        first_norm = np.linalg.norm(first, axis=self.var_axis)
        second_norm = np.linalg.norm(second, axis=self.var_axis)

        # Normalize the input arrays by their respective norms, clipping to avoid division by zero
        first_normed = first / np.expand_dims(first_norm, self.var_axis).clip(min=sensitivity)
        second_normed = second / np.expand_dims(second_norm, self.var_axis).clip(min=sensitivity)

        # Compute the cosine of the angle between first and second using the dot product
        angle_cos = np.sum(first_normed * second_normed, axis=self.var_axis)#.clip(min=-1, max=1)
        # Mask cases where both norms are smaller than the sensitivity threshold (set to NaN)
        angle_cos[(first_norm < sensitivity) & (second_norm < sensitivity)] = np.nan

        # Compute the percentage angular error based on the arccosine of the angle
        return np.arccos(angle_cos) / np.pi * 180
    
class VectorAngle(Metric):
    """
    Computes vector angle
    """
    name = 'angle'
    def __init__(self, degrees=True, normalize=False, arity=1):
        self.degrees = degrees
        self.normalize = normalize
        self._arity = arity

    @property
    def arity(self) -> int:
        return self._arity
    
    def compute(self, *vector_fields, degrees=None, normalize=None, sensitivity=0.1) -> np.ndarray:
        degrees = degrees if degrees is not None else self.degrees
        normalize = normalize if normalize is not None else self.normalize
        res = []
        for vector_field in vector_fields:
            if vector_field.ndim < 3:
                raise ValueError("Vector field must have at least 3 dimensions (e.g., [dir, lat, lon])")
            u, v = np.split(vector_field, 2, axis=-3)  # Split the vector field into u and v components

            angle = np.arctan2(v, u)  # angle in radians, range [-π, π]
            angle[np.linalg.norm(vector_field, axis=-3, keepdims=True) < sensitivity] = np.nan
            if normalize:
                angle = (angle + 2 * np.pi) % (2 * np.pi)

            if degrees:
                angle = np.degrees(angle)
                if normalize:
                    angle = angle % 360
            res.append(angle)
        return _unpack_tuple(res)

class CircularDifference(Metric):
    name = 'circular_difference'
    arity = 2
    def __init__(self, max_value=360):
        self.range = max_value

    def compute(self, a, b):
        diff = a - b
        diff = (diff + self.range / 2) % self.range - self.range / 2
        return diff

class VectorNorm(Metric):
    """
    Computes vector norm
    """
    name = 'norm'
    def __init__(self, arity=1, var_axis=-3, norm=None):
        self._arity = arity
        self.var_axis = var_axis
        self.norm = norm

    @property
    def arity(self) -> int:
        return self._arity
    
    def compute(self, *vector_fields) -> np.ndarray:
        res = []
        for vector_field in vector_fields:
            if vector_field.ndim < 3:
                raise ValueError("Vector field must have at least 3 dimensions (e.g., [dir, lat, lon])")
            norm = np.linalg.norm(vector_field, axis=self.var_axis, ord=self.norm)  # Compute the norm along the vector dimension
            res.append(norm)
        return _unpack_tuple(res)

        
class SequentialMetric(Metric):
    """
    Chains several metrics together:
      out = m_last( ... m_2( m_1(fields...) ) ... )
    """
    def __init__(self, *metrics: Metric):
        if len(metrics) < 2:
            raise ValueError("Need at least two metrics to sequence")
        # the first may be multi‐input; the rest must be unary
        ar = metrics[0].arity
        self._metrics: List[Metric] = list(metrics)
        self._arity = ar

    @property
    def name(self) -> str:
        # e.g. "diff→norm"
        return "→".join(m.name for m in self._metrics)

    @property
    def arity(self) -> int:
        return self._arity

    def compute(self, *fields: np.ndarray) -> np.ndarray:
        if len(fields) != self._arity:
            raise ValueError(f"{self.name} expects {self._arity} inputs, got {len(fields)}")
        # first metric
        out = self._metrics[0].compute(*fields)
        for m in self._metrics[1:]:
            if isinstance(out, (tuple, list)):
                args = list(out)
            else:
                args = [out]
            out = m.compute(*args)
        return out
    
def _unpack_tuple(x):
    """ Unpacks one-element tuples for use as return values """
    if len(x) == 1:
        return x[0]
    else:
        return x
    
class LocalStd(Metric):
    name = 'local_spatial_std'
    arity = 1
    def __init__(self, win_size=21, n_sigma=3.0, eps=1e-8, mask: None | np.ndarray = None, device='cpu'):
        self.win_size = win_size
        self.n_sigma = n_sigma
        self.eps = eps
        self.mask = mask
        self.device = device

    def compute(self, prog: np.ndarray) -> np.ndarray:
        """
        Compute drift success metric using local standard deviation.
        
        Args:
            prog: Prediction array (2, H, W) - [u_component, v_component]
            obs: True state at prediction time (2, H, W)
            obs_init: True state at initial time (2, H, W)
            
        Returns:
            Success mask (H, W) with 1 for successful predictions, 0 for failures,
            and NaN for invalid pixels
        """
        # Create combined validity mask
        mask = np.ones_like(prog[0], dtype=bool) if self.mask is None else self.mask
        valid_mask = (np.isfinite(prog) & mask)
        result = np.full(prog.shape, np.nan, dtype=np.float32)
        # Calculate local standard deviation for both components simultaneously
        std_map = local_std(
            np.nan_to_num(prog[None], nan=0.), 
            np.broadcast_to(valid_mask[None], prog[None].shape),
            win_size=self.win_size,
            n_sigma=self.n_sigma,
            eps=self.eps,
            device=self.device,
        ).squeeze(0)  # (1, 2, H, W)
        print(std_map.shape, valid_mask.shape, result.shape)
        result[valid_mask] = std_map[valid_mask].astype(np.float32)
        return std_map

class SIEvsSICConfusion(Metric):
    """
    Для расчета соответствия однокатегорийных (eg sie) и многокатегорийных (eg sic) 
    данных морского льда.
    eg проверить какому распределению концентраций соответствует 1/8 бальная разметка Sentinel-1 
    """
    name  = "sic_confusion"
    arity = 2

    def __init__(self, to_fraction: bool = False, mask: None | np.ndarray = None):
        """
        to_fraction=True → вход 0…1, False → 0…100
        """
        self.to_fraction = to_fraction
        self.mask = mask

    def _to_decile(self, x):
        if self.to_fraction:
            x = x * 10        # 0…1 → 0…10
        else:
            x = x // 10        # 0…100 → 0…10
        return np.round(x).astype(int)

    def compute(self, sie_reference: np.ndarray, sic_target: np.ndarray) -> np.ndarray:
        values = self._to_decile(sic_target[np.where(sie_reference == 1)])
        return np.histogram(values, bins=np.arange(0, 11))[0]
    
class SicSuccess(Metric):
    """
    Оправдываемость прогноза сплочённости (sea-ice concentration).
    Успех = прогноз и факт лежат в одной стандартной градации по 1/10
            ИЛИ их различие не больше ±1 балла (=10 % концентрации).
    Возвращает 1 / 0 / NaN
    согласно РД 52.27.759-2011:
    7.3.4.1.3 Допустимая ошибка прогнозов сплоченности, разрушения льдов,
    торосистости, сжатия льдов и других показателей состояния ледового режима,
    определяемых в баллах, принимается равной ±1 балл.
    """
    name  = "sic_success"
    arity = 2

    def __init__(self, to_fraction: bool = False, mask: None | np.ndarray = None):
        """
        to_fraction=True → вход 0…1, False → 0…100
        """
        self.to_fraction = to_fraction
        self.mask = mask

    def _to_decile(self, x):
        if self.to_fraction:
            x = x * 10        # 0…1 → 0…10
        else:
            x = x // 10        # 0…100 → 0…10
        return np.round(x).astype(int)

    def compute(self, prog: np.ndarray, obs: np.ndarray) -> np.ndarray:
        mask = np.ones_like(prog, dtype=bool) if self.mask is None else self.mask
        mask = np.isfinite(prog) & np.isfinite(obs) & mask
        out  = np.full_like(prog, np.nan, dtype=float)

        p_dec = self._to_decile(prog[mask])
        o_dec = self._to_decile(obs[mask])
        hit   = np.abs(p_dec - o_dec) <= 1
        out[mask] = hit.astype(float)
        return out
    

class CategoricalSicSuccess(SicSuccess):
    """
    Категориальная оправдываемость прогноза сплочённости (sea-ice concentration).

    Успех = и прогноз, и факт относятся к одной и той же категории сплочённости:

      0 баллов:  Чистая вода (нет льда).
      1-3 балла: Редкий лед.
      4-6 баллов: Разреженный лед.
      7-8 баллов: Сплоченный лед.
      9-10 баллов: Очень сплоченный / сплошной лед.

    Возвращает поле из 1 / 0 / NaN.
    NaN — там, где входные данные не заданы или не попали ни в одну категорию.
    """
    name  = "sic_categorical_success"
    arity = 2

    def __init__(
        self,
        to_fraction: bool = False,
        mask: None | np.ndarray = None,
        categories: Optional[List[Tuple[int, int]]] = None,
    ):
        """
        to_fraction=True → вход 0…1, False → 0…100.

        categories — список диапазонов в баллах (включительно).
        По умолчанию используются:
            (0, 0), (1, 3), (4, 6), (7, 8), (9, 10)
        """
        super().__init__(to_fraction=to_fraction, mask=mask)
        if categories is None:
            categories = [
                (0, 0),   # чистая вода
                (1, 3),   # редкий лед
                (4, 6),   # разреженный лед
                (7, 8),   # сплоченный лед
                (9, 10),  # очень сплоченный / сплошной лед
            ]
        self.categories = categories

    def _to_category(self, deciles: np.ndarray) -> np.ndarray:
        """
        Перевод баллов (0-10) в номер категории (0, 1, 2, ...).
        Если балл не попал ни в одну категорию — возвращается -1.
        """
        cat = np.full_like(deciles, -1, dtype=int)
        for idx, (lo, hi) in enumerate(self.categories):
            m = (deciles >= lo) & (deciles <= hi)
            cat[m] = idx
        return cat

    def compute(self, prog: np.ndarray, obs: np.ndarray) -> np.ndarray:
        # базовая маска по данным + внешняя маска, как в SicSuccess
        mask = np.ones_like(prog, dtype=bool) if self.mask is None else self.mask
        mask = np.isfinite(prog) & np.isfinite(obs) & mask

        out = np.full_like(prog, np.nan, dtype=float)

        # Перевод в баллы 0–10 с учетом to_fraction
        p_dec = self._to_decile(prog[mask])
        o_dec = self._to_decile(obs[mask])

        # Перевод баллов в категории
        p_cat = self._to_category(p_dec)
        o_cat = self._to_category(o_dec)

        # Отбрасываем точки, не попавшие ни в одну категорию
        valid = (p_cat >= 0) & (o_cat >= 0)

        # Успех = одна и та же категория
        hit_valid = (p_cat == o_cat) & valid

        # Собираем локальный результат и возвращаем в исходную сетку
        local_out = np.full(p_dec.shape, np.nan, dtype=float)
        local_out[valid] = hit_valid[valid].astype(float)

        out[mask] = local_out
        return out

class DriftSuccess(Metric):
    name = "drift_success"
    arity = 3  # prog, obs, obs_init

    def __init__(self, win_size=21, n_sigma=3.0, eps=1e-8, mask: None | np.ndarray = None, min_tolerance: float = 0., device='cpu'):
        self.win_size = win_size
        self.n_sigma = n_sigma
        self.eps = eps
        self.mask = mask
        self.min_tolerance = min_tolerance
        self.device = device

    def compute(self, prog: np.ndarray, obs: np.ndarray, obs_init: np.ndarray) -> np.ndarray:
        """
        Compute drift success metric using local standard deviation.
        
        Args:
            prog: Prediction array (2, H, W) - [u_component, v_component]
            obs: True state at prediction time (2, H, W)
            obs_init: True state at initial time (2, H, W)
            
        Returns:
            Success mask (H, W) with 1 for successful predictions, 0 for failures,
            and NaN for invalid pixels
        """
        # Calculate true state evolution
        true_change = obs - obs_init  # (2, H, W)
        
        # Create combined validity mask
        mask = np.ones_like(prog[0], dtype=bool) if self.mask is None else self.mask
        valid_mask = (
            np.isfinite(prog).all(axis=0) & 
            np.isfinite(obs).all(axis=0) & 
            np.isfinite(obs_init).all(axis=0) &
            mask)
        
        # Initialize output with NaNs
        result = np.full(prog.shape[1:], np.nan, dtype=np.float32)
        
        # Skip processing if no valid pixels
        if not valid_mask.any():
            print('No valid values in data')
            return result
        
        # Calculate prediction error
        error = prog - obs  # (2, H, W)
        
        # Convert to PyTorch tensors
        true_change = true_change[None]  # (1, 2, H, W)  # todo can do without torch
        mask_t = ~np.isnan(true_change)  # (1, 2, H, W)
        
        # Calculate local standard deviation for both components simultaneously
        std_map = local_std(
            np.nan_to_num(true_change, nan=0.), 
            mask_t,
            win_size=self.win_size,
            n_sigma=self.n_sigma,
            eps=self.eps,
            device=self.device,
        )  # (1, 2, H, W)
        
        # Convert back to numpy
        std_map = np.linalg.norm(std_map.squeeze(0), axis=0)  # (1, H, W)
        
        # Calculate threshold (±0.674σ)
        threshold = np.maximum(0.674 * std_map, self.min_tolerance)  # (1, H, W)
        # Check if errors are within threshold for both components
        success = (np.linalg.norm(error, axis=0) <= threshold)  # (1, H, W)
        
        # Both components must be within threshold for success
        # success = within_threshold.all(axis=0)  # (H, W)
        
        # Set valid pixels with 1 for success, 0 for failure
        result[valid_mask] = success[valid_mask].astype(np.float32)
        return result


class ThickSuccess(Metric):
    """
    Оправдываемость прогноза толщины льда.
    Успех, если |prog - obs| ≤ 0.3 * |obs - obs_init|
    согласно РД 52.27.759-2011:
    7.3.4.1.2 При оценке прогнозов толщины льда допустимая ошибка прогноза
    устанавливается равной 30 % от фактического ее изменения за период
    заблаговременности прогноза (таблица 8).
    Фактическое изменение
    _______________________________________________________________________
    | толщины льда, см      | > 10 | 11–15 | 16–20 | 21–25 | 26–30 | > 30 |
    | Допустимая ошибка, см |  ± 3 |  ± 4  | ± 6   | ± 7   |  ± 8  | ± 10 |
    -----------------------------------------------------------------------
    """
    name  = "sit_success"
    arity = 3          # prog, obs, obs_init

    def __init__(self, sensitivity: float = 0.3, min_tolerance: float = 0.1, mask: None | np.ndarray = None):
        self.sensitivity = sensitivity 
        self.mask = mask
        self.min_tolerance = min_tolerance

    def compute(self, prog, obs, obs_init):
        mask = np.ones_like(prog, dtype=bool) if self.mask is None else self.mask
        mask = np.isfinite(prog) & np.isfinite(obs) & mask
        out  = np.full_like(prog, np.nan, dtype=float)

        change = np.abs(obs - obs_init)
        hit = np.abs(prog - obs) <= self.sensitivity * np.maximum(change, self.min_tolerance)

        out[mask] = hit[mask].astype(float)
        return out
    

def efficiency_wrapper(success_cls):
    """
    Decorator that converts a “success”-type metric class into its
    **efficiency** counterpart:
    
        efficiency = success(prog, obs, obs_init) \
                   – success(obs_init, obs, obs_init)
                   
    The new class:
    * inherits all arguments/attributes of the wrapped class;
    * keeps the same constructor signature (via normal subclassing);
    * forces ``arity = 3`` because an inertial forecast is always needed;
    * works for both 2- and 3-argument ``compute`` implementations.
    для получения действительной эффективности метода необходимо
    из его обеспеченности вычесть действительную обеспеченность климатического/инернционного прогноза
    """
    class EfficiencyMetric(success_cls):
        # rename e.g. "sic_success" → "sic_efficiency"
        name = getattr(success_cls, "name", success_cls.__name__).replace(
            "_success", "_efficiency"
        )
        arity = 3
        def compute(self, prog, obs, obs_init, *extra, **kw):
            """
            Args
            ----
            prog      : model forecast  at verification time
            obs       : verifying analysis/observation
            obs_init  : state at initial time (persistence forecast)
            *extra    : any extra positional args required by the base metric
            **kw      : any keyword args required by the base metric
            """
            if success_cls.arity == 2:
                succ_forecast  = super().compute(prog, obs, *extra, **kw)
                succ_persist   = super().compute(obs_init, obs, *extra, **kw)
            else:  # arity == 3 (e.g. drift, thickness)
                succ_forecast  = super().compute(prog, obs, obs_init, *extra, **kw)
                succ_persist   = super().compute(obs_init, obs, obs_init, *extra, **kw)

            return succ_forecast - succ_persist

    # make the wrapper look like the original class in repr / help()
    EfficiencyMetric.__name__ = success_cls.__name__.replace(
        "Success", "Efficiency"
    )
    EfficiencyMetric.__qualname__ = EfficiencyMetric.__name__
    EfficiencyMetric.__doc__ = (
        f"{success_cls.__doc__}\n\n"
        "Wrapped as an efficiency metric. See efficiency_wrapper()."
    )
    return EfficiencyMetric


class GreatCircleDistance(Metric):
    """
    Computes the great circle distance between two points on a sphere given their latitudes and longitudes.
    The input arrays should have the last dimension of size 2, where the first element is latitude and the second is longitude.
    """
    name = 'great_circle_distance'
    arity = 2
    _WGS84_GEOD = Geod(ellps="WGS84")
    def compute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        lat1 = a[..., 0]
        lon1 = a[..., 1]
        lat2 = b[..., 0]
        lon2 = b[..., 1]
        az_deg, _, dist_m = self._WGS84_GEOD.inv(lon1, lat1, lon2, lat2)
        return dist_m
    