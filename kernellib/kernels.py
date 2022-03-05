from sklearn.metrics.pairwise import rbf_kernel
from sklearn.gaussian_process.kernels import (
    StationaryKernelMixin,
    NormalizedKernelMixin,
    Kernel,
    Hyperparameter,
    Product,
)
from kernellib.utils import add_dim
import numpy as np


class SpatioTemporalKernel(Product):
    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)
        Y : array-like of shape (n_samples_Y, n_features) or list of object,\
            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.
        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        spatial_coords, temporal_coords = self._split_inputs(X, Y)

        if eval_gradient:
            K1, K1_gradient = self.k1(
                spatial_coords[0], spatial_coords[1], eval_gradient=True
            )
            K2, K2_gradient = self.k2(
                temporal_coords[0], temporal_coords[1], eval_gradient=True
            )
            return K1 * K2, np.dstack(
                (K1_gradient * K2[:, :, np.newaxis], K2_gradient * K1[:, :, np.newaxis])
            )
        else:
            return self.k1(spatial_coords[0], spatial_coords[1]) * self.k2(
                temporal_coords[0], temporal_coords[1]
            )

    def _split_inputs(self, X, Y):
        return split_coordinates(X, Y)


def split_coordinates(x_coords, y_coords=None):

    n_samples, n_dims = x_coords.shape

    # extract temporal coords
    if y_coords is None:
        y_coords = x_coords

    X_temporal, Y_temporal = x_coords[:, 0], y_coords[:, 0]

    # add extra dimension
    X_temporal, Y_temporal = add_dim(X_temporal), add_dim(Y_temporal)

    # extract spatial coords
    X_spatial, Y_spatial = x_coords[:, 1:], y_coords[:, 1:]

    assert X_temporal.ndim == Y_temporal.ndim == 2
    assert X_temporal.shape[1] == Y_temporal.shape[1] == 1
    assert X_spatial.ndim == X_spatial.ndim == 2
    assert X_spatial.shape[1] == Y_spatial.shape[1] == 2

    return (X_temporal, Y_temporal), (X_spatial, Y_spatial)


# def temporal_gauss_kernel(x_time_coords, y_time_coords, time_scale):
#     gamma = 1 / time_scale**2
#     x_time_coords = add_dim(x_time_coords)
#     y_time_coords = add_dim(y_time_coords)

#     k = rbf_kernel(x_time_coords, y_time_coords, gamma=gamma)
#     return k


# def spatial_gauss_kernel(x_spatial_coords, y_spatial_coords, spatial_scale):
#     gamma = 1 / spatial_scale**2
#     x_spatial_coords = add_dim(x_spatial_coords)
#     y_spatial_coords = add_dim(y_spatial_coords)
#     k = rbf_kernel(x_spatial_coords, y_spatial_coords, gamma=gamma)
#     return k


# def spatiotemporal_kernel(x_coords, y_coords, time_scale, spatial_scale):
#     # print(x_coords.shape, y_coords.shape)
#     k1 = temporal_gauss_kernel(x_coords[:, 0], y_coords[:, 0], time_scale)
#     k2 = spatial_gauss_kernel(x_coords[:, 1:], y_coords[:, 1:], spatial_scale)

#     return k1 * k2


# class SpatioTemporalKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
#     def __init__(
#         self,
#         spatial_length_scale=1.0,
#         temporal_length_scale=1.0,
#         spatial_length_scale_bounds=(1e-5, 1e5),
#         temporal_length_scale_bounds=(1e-5, 1e5),
#     ):
#         self.spatial_length_scale = spatial_length_scale
#         self.spatial_length_scale_bounds = spatial_length_scale_bounds
#         self.temporal_length_scale = temporal_length_scale
#         self.temporal_length_scale_bounds = temporal_length_scale_bounds

#     @property
#     def hyperparameter_spatial_length_scale(self):
#         """Returns the length scale"""
#         return Hyperparameter(
#             "spatial_length_scale", "numeric", self.spatial_length_scale_bounds
#         )

#     @property
#     def hyperparameter_temporal_length_scale(self):
#         """Returns the length scale"""
#         return Hyperparameter(
#             "temporal_length_scale", "numeric", self.temporal_length_scale_bounds
#         )

#     def __call__(self, X, Y=None, eval_gradient=False):

#         if Y is None:
#             K = spatiotemporal_kernel(
#                 X, X, self.temporal_length_scale, self.spatial_length_scale
#             )
#         else:
#             K = spatiotemporal_kernel(
#                 X, Y, self.temporal_length_scale, self.spatial_length_scale
#             )

#         if eval_gradient:
#             raise NotImplementedError()

#         else:
#             return K
