# # computer kernel matrix
# y = obs_data.data[:, None]
# alpha = noise ** 2
# copy = True
# sample_weight = None

# # compute weights
# K = kernel(obs_coords, obs_coords)
# np.fill_diagonal(K, K.diagonal() + alpha)

# # compute kernel
# weights, L = cholesky_solve(K, y)

# # CHECK FOR EQUIVALENCY
# K = kernel(obs_coords, obs_coords)
# weights_ = _solve_cholesky_kernel(K, y, alpha=alpha, sample_weight=sample_weight, copy=copy)
# np.testing.assert_array_almost_equal(weights_, weights)
