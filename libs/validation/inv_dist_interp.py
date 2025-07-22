import torch
import numpy as np
from sklearn.neighbors import KDTree, BallTree


class InvDistTree(torch.nn.Module):
    def __init__(self, x, q, leaf_size=10, n_near=6, sigma_squared=None,
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
        res = (z[..., self.ix] * self.weights).sum(-1)
        return res

    def calc_input_tensor_mask(self, mask_shape, distance_criterion=0.15, fill_value=0):
        s = mask_shape
        assert s[-1] * s[-2] == self.distances.shape[0], "mask shape should be compatible with calculated distances"
        mask = torch.ones([s[-1] * s[-2]])
        mask[np.where(self.distances.mean(-1) > distance_criterion)] = fill_value
        mask = mask.reshape(*s).to(self.device)
        return mask


def gauss_function(x, sigma_squared=1):
    if isinstance(x, np.ndarray):
        x_torch = torch.from_numpy(x)
    else:
        x_torch = x
    f_x = 1 / np.sqrt(2*np.pi*sigma_squared) * torch.exp(-0.5 * x_torch * x_torch / sigma_squared)
    return f_x


class InvDistTree_np():
    def __init__(self, x, q, leaf_size=10, n_near=6, sigma_squared=None,
                 distance_metric='euclidean', inv_dist_mode='gaussian'):
        super().__init__()

        self.ix = None
        self.weights = None
        self.distances = None
        self.dist_mode = inv_dist_mode
        self.x = np.asarray(x)
        self.q = np.asarray(q)
        self.k = 1
        self.leaf_size = leaf_size
        self.tree = self.build_tree(distance_metric)  # KDTree(x, leafsize=leaf_size)  # build the tree
        self.calc_interpolation_weights(n_near, sigma_squared)

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
        self.weights = self.weights / np.sum(self.weights, axis=-1, keepdims=True)
        self.weights = np.nan_to_num(self.weights, nan=1/n_near)
        self.weights = self.weights.astype(float)

    def calc_dist_coefs(self, dist, sigma_squared=None):
        if self.dist_mode == 'inverse':
            return 1 / dist
        elif self.dist_mode == 'gaussian':
            sigma_squared = sigma_squared if sigma_squared else np.square(np.median(self.distances)) / 9 / self.k
            return gauss_function_np(dist, sigma_squared=sigma_squared)

    def __call__(self, z):
        res = (z[..., self.ix] * self.weights).sum(-1)
        return res

    def calc_input_tensor_mask(self, mask_shape, distance_criterion=0.15, fill_value=0):
        s = mask_shape
        assert s[-1] * s[-2] == self.distances.shape[0], "mask shape should be compatible with calculated distances"
        mask = np.ones([s[-1] * s[-2]])
        mask[np.where(self.distances.mean(-1) > distance_criterion)] = fill_value
        mask = mask.reshape(*s)
        return mask

def gauss_function_np(x, sigma_squared=1):
    f_x = 1 / np.sqrt(2*np.pi*sigma_squared) * np.exp(-0.5 * x * x / sigma_squared)
    return f_x