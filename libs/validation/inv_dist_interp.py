import torch
import numpy as np
from sklearn.neighbors import KDTree, BallTree


class InvDistTree(torch.nn.Module):
    def __init__(self, x, q, leaf_size=10, n_near=6, sigma_squared=None, has_nans=False,
                 distance_metric='euclidean', inv_dist_mode='gaussian', device='cpu'):
        super().__init__()

        self.ix = None
        self.weights = None
        self.distances = None
        self.dist_mode = inv_dist_mode
        self.x = np.asarray(x)
        self.q = np.asarray(q)
        self.k = 1
        self.device = device
        self.leaf_size = leaf_size
        self.tree = self.build_tree(distance_metric)  # KDTree(x, leafsize=leaf_size)  # build the tree
        self.calc_interpolation_weights(n_near, sigma_squared)
        self.nan_sustainable = has_nans
        self.to(device)

    def build_tree(self, distance_metric):
        if distance_metric == 'euclidean':
            self.tree = KDTree(self.x, leaf_size=self.leaf_size)
        elif distance_metric == 'haversine':
            self.q = np.radians(self.q)
            self.tree = BallTree(np.radians(self.x), leaf_size=self.leaf_size, metric=distance_metric)
        else:
            raise NotImplementedError
        return self.tree

    def calc_interpolation_weights(self, n_near=6, sigma_squared=None):
        self.distances, self.ix = self.tree.query(self.q, k=n_near)
        if n_near == 1:
            self.distances = self.distances[:, None]
            self.ix = self.ix[:, None]
        if np.where(self.distances < 1e-10)[0].size != 0:
            print('Zeros in indices!')
        self.weights = self.calc_dist_coefs(self.distances, sigma_squared)
        self.weights = self.weights / torch.sum(self.weights, dim=-1, keepdim=True)
        self.weights = torch.nan_to_num(self.weights, 1/n_near)
        self.weights = self.weights.type(torch.float).to(self.device)

    def calc_dist_coefs(self, dist, sigma_squared=None):
        if self.dist_mode == 'inverse':
            return torch.from_numpy(1 / dist)
        elif self.dist_mode == 'gaussian':
            sigma_squared = sigma_squared if sigma_squared else np.square(np.median(self.distances)) / 9 / self.k
            return gauss_function(dist, sigma_squared=sigma_squared)
        elif self.dist_mode == 'LinearNN':  # todo
            raise NotImplementedError

    def __call__(self, z):
        if self.nan_sustainable:
            return self._nan_sustainable_interp(z)
        else:
            res = (z[..., self.ix] * self.weights).sum(-1)
        return res

    def calc_input_tensor_mask(self, mask_shape, distance_criterion=0.15, fill_value=0):
        s = mask_shape
        assert s[-1] * s[-2] == self.distances.shape[0], "mask shape should be compatible with calculated distances"
        mask = torch.ones([s[-1] * s[-2]])
        mask[np.where(self.distances.mean(-1) > distance_criterion)] = fill_value
        mask = mask.reshape(*s).to(self.device)
        return mask

    def _nan_sustainable_interp(self, z):
        """
        Interpolate using inverse distance weighting, ignoring NaN values in `z`.

        Args:
            z (torch.Tensor): Input tensor of shape [..., N], where N is the number of data points.

        Returns:
            torch.Tensor: Interpolated values of shape [..., num_query_points].
        """
        # Gather neighbor values: [..., num_query_points, n_near]
        z_gathered = z[..., self.ix]

        # Create a mask for non-NaN values
        valid_mask = ~torch.isnan(z_gathered)  # [..., num_query_points, n_near]

        # Set NaNs to zero (to avoid affecting the sum)
        z_gathered = torch.where(valid_mask, z_gathered, torch.tensor(0.0, device=z_gathered.device))

        # Zero out weights where values are NaN
        weights_masked = torch.where(valid_mask, self.weights, torch.tensor(0.0, device=self.weights.device))

        # Re-normalize weights to sum to 1 (avoid division by zero)
        weight_sums = weights_masked.sum(dim=-1, keepdim=True)
        weights_normalized = torch.where(weight_sums > 0, weights_masked / weight_sums, weights_masked)

        # Compute weighted sum
        result = (z_gathered * weights_normalized).sum(dim=-1)

        # If all neighbors were NaN, result should be NaN
        result = torch.where(weight_sums.squeeze(-1) > 0, result, torch.tensor(float('nan'), device=result.device))

        return result


def gauss_function(x, sigma_squared=1):
    if isinstance(x, np.ndarray):
        x_torch = torch.from_numpy(x)
    else:
        x_torch = x
    f_x = 1 / np.sqrt(2*np.pi*sigma_squared) * torch.exp(-0.5 * x_torch * x_torch / sigma_squared)
    return f_x


# class InvDistTree_np():
#     def __init__(self, x, q, leaf_size=10, n_near=6, sigma_squared=None,
#                  distance_metric='euclidean', inv_dist_mode='gaussian'):
#         super().__init__()

#         self.ix = None
#         self.weights = None
#         self.distances = None
#         self.dist_mode = inv_dist_mode
#         self.x = np.asarray(x)
#         self.q = np.asarray(q)
#         self.k = 1
#         self.leaf_size = leaf_size
#         self.tree = self.build_tree(distance_metric)  # KDTree(x, leafsize=leaf_size)  # build the tree
#         self.calc_interpolation_weights(n_near, sigma_squared)

#     def build_tree(self, distance_metric):
#         if distance_metric == 'euclidean':
#             self.tree = KDTree(self.x, leaf_size=self.leaf_size)
#         elif distance_metric == 'haversine':
#             self.q = np.radians(self.q)
#             self.tree = BallTree(np.radians(self.x), leaf_size=self.leaf_size, metric=distance_metric)
#         else:
#             raise NotImplementedError
#         return self.tree

#     def calc_interpolation_weights(self, n_near=6, sigma_squared=None):
#         self.distances, self.ix = self.tree.query(self.q, k=n_near)
#         if n_near == 1:
#             self.distances = self.distances[:, None]
#             self.ix = self.ix[:, None]
#         if np.where(self.distances < 1e-10)[0].size != 0:
#             print('Zeros in indices!')
#         self.weights = self.calc_dist_coefs(self.distances, sigma_squared)
#         self.weights = self.weights / np.sum(self.weights, axis=-1, keepdims=True)
#         self.weights = np.nan_to_num(self.weights, nan=1/n_near)
#         self.weights = self.weights.astype(float)

#     def calc_dist_coefs(self, dist, sigma_squared=None):
#         if self.dist_mode == 'inverse':
#             return 1 / dist
#         elif self.dist_mode == 'gaussian':
#             sigma_squared = sigma_squared if sigma_squared else np.square(np.median(self.distances)) / 9 / self.k
#             return gauss_function_np(dist, sigma_squared=sigma_squared)

#     def __call__(self, z):
#         res = (z[..., self.ix] * self.weights).sum(-1)
#         return res

#     def calc_input_tensor_mask(self, mask_shape, distance_criterion=0.15, fill_value=0):
#         s = mask_shape
#         assert s[-1] * s[-2] == self.distances.shape[0], "mask shape should be compatible with calculated distances"
#         mask = np.ones([s[-1] * s[-2]])
#         mask[np.where(self.distances.mean(-1) > distance_criterion)] = fill_value
#         mask = mask.reshape(*s)
#         return mask

def gauss_function_np(x, sigma_squared=1):
    f_x = 1 / np.sqrt(2*np.pi*sigma_squared) * np.exp(-0.5 * x * x / sigma_squared)
    return f_x


def gauss_function_np(x, sigma_squared=1.0):
    return (1.0 / np.sqrt(2.0 * np.pi * sigma_squared)) * np.exp(-0.5 * x * x / sigma_squared)

class InvDistTree_np:
    """
    Inverse-distance / Gaussian-distance interpolator supporting:
      - static queries: q shape (..., 2)
      - time-varying queries: q shape (T, ..., 2) aligned with z's time axis
    """

    def __init__(
        self,
        x,
        q=None,
        *,
        q_time_axis=None,
        leaf_size=10,
        n_near=6,
        sigma_squared=None,
        distance_metric="euclidean",
        inv_dist_mode="gaussian",
        dtype=np.float32,
    ):
        self.dist_mode = inv_dist_mode
        self.leaf_size = int(leaf_size)
        self.metric = distance_metric
        self.dtype = dtype

        self.x = np.asarray(x, dtype=float)
        self.tree = self._build_tree(self.x, self.metric)

        self.k = None
        self.q_shape = None
        self.ix = None
        self.distances = None
        self.weights = None

        if q is not None:
            self.set_queries(q, n_near=n_near, sigma_squared=sigma_squared, q_time_axis=q_time_axis)

    def _build_tree(self, x, metric):
        if metric == "euclidean":
            return KDTree(x, leaf_size=self.leaf_size)
        if metric == "haversine":
            # BallTree haversine expects radians, ordered as [lat, lon]
            return BallTree(np.radians(x), leaf_size=self.leaf_size, metric="haversine")
        raise NotImplementedError(f"Unknown distance_metric={metric}")

    def set_queries(self, q, *, q_time_axis=None, n_near=6, sigma_squared=None):
        q = np.asarray(q)
        if q.shape[-1] != 2:
            raise ValueError(f"Expected q[...,2] last dim=2 (lat,lon). Got {q.shape}")

        self.k = int(n_near)

        if q_time_axis is None:
            # static: q shape (...,2) -> flat (Q,2)
            self.time_varying = False
            self.q_shape = q.shape[:-1]
            self.q_tail_shape = self.q_shape
            q2 = q.reshape(-1, 2)
            self.T = None
        else:
            # time-varying: move time axis to front
            self.time_varying = True
            q_time_axis = q_time_axis % q.ndim
            q_tm = np.moveaxis(q, q_time_axis, 0)      # (T, ..., 2)
            self.T = q_tm.shape[0]
            self.q_shape = q_tm.shape[:-1]
            self.q_tail_shape = q_tm.shape[1:-1]     # e.g. (N,)
            q2 = q_tm.reshape(-1, 2)                 # (T*Q, 2)

        # radians only for haversine queries
        if self.metric == "haversine":
            q2 = np.radians(q2)

        distances, ix = self.tree.query(q2, k=self.k)

        if self.k == 1:
            distances = distances.reshape(-1, 1)
            ix = ix.reshape(-1, 1)

        # reshape back
        if not self.time_varying:
            self.distances = distances.reshape(*self.q_tail_shape, self.k)
            self.ix = ix.reshape(*self.q_tail_shape, self.k)
        else:
            Q = int(np.prod(self.q_tail_shape)) if self.q_tail_shape else 1
            self.distances = distances.reshape(self.T, Q, self.k)
            self.ix = ix.reshape(self.T, Q, self.k)

        # Weights
        if self.k == 1:
            w = np.ones_like(self.distances, dtype=float)
        else:
            w = self._calc_dist_coefs(self.distances, sigma_squared=sigma_squared)
            w_sum = np.sum(w, axis=-1, keepdims=True)
            # avoid divide-by-zero if all weights became 0
            w = np.divide(w, w_sum, out=np.zeros_like(w), where=(w_sum != 0))

            # handle exact hits (distance ~ 0): force one-hot on the nearest
            zero_hit = (self.distances < 1e-12)
            if np.any(zero_hit):
                # for each query, if any zero, set that one to 1, others 0
                first_zero = np.argmax(zero_hit, axis=-1)  # index of first True (or 0 if none)
                has_zero = np.any(zero_hit, axis=-1)
                w[...] = np.where(has_zero[..., None], 0.0, w)
                # scatter 1.0 at first_zero
                it = np.nditer(has_zero, flags=["multi_index"])
                for hz in it:
                    if bool(hz):
                        mi = it.multi_index
                        w[mi + (first_zero[mi],)] = 1.0

        self.weights = w.astype(self.dtype, copy=False)

    def _calc_dist_coefs(self, dist, sigma_squared=None):
        if self.dist_mode == "inverse":
            return 1.0 / np.maximum(dist, 1e-12)
        if self.dist_mode == "gaussian":
            if sigma_squared is None:
                # heuristic based on median distance over all queries/neighbors
                med = np.median(dist.astype(float))
                sigma_squared = (med * med) / 9.0
                sigma_squared = max(sigma_squared, 1e-12)
            return gauss_function_np(dist, sigma_squared=float(sigma_squared))
        raise NotImplementedError(f"Unknown inv_dist_mode={self.dist_mode}")

    def interpolate_static(self, z, *, space_axis=-1):
        """
        Static queries: q_shape = (...). z is (..., S) along space_axis.
        Output: z with space_axis removed, query dims appended at the end.
        """
        if self.ix is None:
            raise RuntimeError("Call set_queries(q, ...) first.")

        z = np.asarray(z)
        z2 = np.moveaxis(z, space_axis, -1)  # (..., S)
        S = z2.shape[-1]

        # Flatten query dims -> Q
        ix2 = self.ix.reshape(-1, self.k)
        w2 = self.weights.reshape(-1, self.k)
        Q = ix2.shape[0]

        # Broadcast z to (..., Q, S) as a view (no big copy)
        zq = np.broadcast_to(z2[..., None, :], z2.shape[:-1] + (Q, S))
        idx = np.broadcast_to(ix2, z2.shape[:-1] + ix2.shape)

        gathered = np.take_along_axis(zq, idx, axis=-1)  # (..., Q, K)
        w_full = np.broadcast_to(w2, gathered.shape)
        out = np.sum(gathered * w_full, axis=-1)  # (..., Q)
        out = out.reshape(z2.shape[:-1] + self.q_shape)  # (..., *q_shape)
        return out

    def interpolate_timevarying(self, z, *, time_axis, space_axis=-1):
        """
        Time-varying queries: q must be (T, ..., 2), so q_shape = (T, ...).
        z must have a matching time dimension (same T) at time_axis, and space at space_axis.

        Returns: z with space_axis removed, and query "tail" dims appended,
                 while preserving original axis order (except space removed).
        Common case: z (V,T,S), q (T,N,2) -> out (V,T,N)
        """
        if self.ix is None:
            raise RuntimeError("Call set_queries(q, ...) first.")
        if len(self.q_shape) < 1:
            raise RuntimeError("q_shape invalid.")
        z = np.asarray(z)

        # Normalize axes
        time_axis = time_axis % z.ndim
        space_axis = space_axis % z.ndim

        # Move time & space to the end for computation: (...other..., T, S)
        z2 = np.moveaxis(z, (time_axis, space_axis), (-2, -1))
        T = z2.shape[-2]
        S = z2.shape[-1]
        if self.q_shape[0] != T:
            raise ValueError(f"q has T={self.q_shape[0]} but z has T={T} along time_axis={time_axis}")

        # Flatten query tail dims -> Q
        q_tail_shape = self.q_shape[1:]          # e.g. (N,)
        Q = int(np.prod(q_tail_shape)) if q_tail_shape else 1

        ix2 = self.ix.reshape(T, Q, self.k)
        w2 = self.weights.reshape(T, Q, self.k)

        # Create broadcast views for take_along_axis
        # z2: (..., T, S) -> (..., T, Q, S)
        ztq = np.broadcast_to(z2[..., :, None, :], z2.shape[:-2] + (T, Q, S))
        idx = np.broadcast_to(ix2, z2.shape[:-2] + ix2.shape)

        gathered = np.take_along_axis(ztq, idx, axis=-1)  # (..., T, Q, K)
        w_full = np.broadcast_to(w2, gathered.shape)
        out2 = np.sum(gathered * w_full, axis=-1)         # (..., T, Q)
        out2 = out2.reshape(z2.shape[:-2] + (T,) + q_tail_shape)  # (..., T, *tail)

        # Reorder axes back to: original z axes except space removed, then query tail dims appended.
        axes_without_space = [ax for ax in range(z.ndim) if ax != space_axis]
        other_axes = [ax for ax in range(z.ndim) if ax not in (time_axis, space_axis)]
        pos_in_other = {ax: i for i, ax in enumerate(other_axes)}
        time_pos_in_out2 = len(other_axes)  # in out2, time is right after "other_axes"

        # Build permutation for the non-query part
        perm = []
        for ax in axes_without_space:
            if ax == time_axis:
                perm.append(time_pos_in_out2)
            else:
                perm.append(pos_in_other[ax])

        # Append query tail dims (they are already at the end in out2)
        perm += list(range(time_pos_in_out2 + 1, out2.ndim))

        out = np.transpose(out2, perm)
        return out
    
    def __call__(self, z, *, time_axis=None, space_axis=-1):
        """
        Interpolate values `z` using precomputed indices/weights.

        - If queries are static (set_queries(time_axis=None)), calls interpolate_static.
          `time_axis` is ignored.

        - If queries are time-varying (set_queries(time_axis=...)), calls interpolate_timevarying.
          You MUST provide `time_axis` for `z` so it aligns with q's time dimension.

        Parameters
        ----------
        z : array-like
            Data to interpolate. Must include a spatial axis (space_axis).
            For time-varying queries, must also include time axis matching q's T.
        time_axis : int | None
            Time axis of `z` (required only for time-varying queries).
        space_axis : int
            Spatial axis of `z` corresponding to flattened source points S (e.g. H*W).

        Returns
        -------
        np.ndarray
            Interpolated array. For common case:
              z (V,T,S) + q (T,N,2) -> out (V,T,N)
        """
        if self.ix is None or self.weights is None:
            raise RuntimeError("Interpolator has no queries set. Call set_queries(q, ...) first.")

        if not getattr(self, "time_varying", False):
            # static q: (...,2)
            return self.interpolate_static(z, space_axis=space_axis)

        # time-varying q: (T,...,2)
        if time_axis is None:
            raise ValueError(
                "This interpolator was built with time-varying queries; "
                "please pass time_axis for z (e.g. time_axis=1 for z shape (V,T,S))."
            )
        return self.interpolate_timevarying(z, time_axis=time_axis, space_axis=space_axis)