import numpy as np
import matplotlib.pyplot as plt
import utils as uu


# range of values
x_lim = [0, 20]

# train inputs
n_train = 10
# np.random.seed(10)
x_train = np.random.uniform(x_lim[0], x_lim[1], n_train)
f_train = uu.objective(x_train)
np.random.seed(None)

# test inputs
n_test = 200
x_test = np.linspace(x_lim[0], x_lim[1], n_test)

# predictive distribution
n_samples = 3
scale = 1
cov_test_test = uu.se_matrix(x_test, scale=scale)
cov_train_train = uu.se_matrix(x_train, scale=scale)
cov_test_train = uu.se_matrix(x_test, x_train, scale=scale)
cov_train_test = cov_test_train.T
cov_train_train_inv = np.linalg.inv(cov_train_train)

mu = np.dot(np.dot(cov_test_train, cov_train_train_inv), f_train)
cov = cov_test_test - np.dot(np.dot(cov_test_train, cov_train_train_inv), cov_train_test)
# cov += 1e-6 * np.eye(n_test)  # avoid singularities

# pick samples
samples = np.random.multivariate_normal(mu, cov, n_samples)

# FIGURES
# plot objective function
t = np.linspace(x_lim[0], x_lim[1], 100)
plt.plot(t, uu.objective(t), c='b')

# plot train inputs
plt.scatter(x_train, f_train, marker='+', c='k', s=120, zorder=10)

# plot samples
# for i in range(n_samples):
#     plt.plot(x_test, samples[i, :], marker='')

# plot mean +- 2*variance
plt.plot(x_test, mu, c='k', linewidth=0.8)
plt.fill_between(x_test, mu - 2 * np.sqrt(cov[np.diag_indices_from(cov)]),
                 mu + 2 * np.sqrt(cov[np.diag_indices_from(cov)]), facecolor='gray', alpha=0.5)

plt.grid(alpha=0.5)
plt.xlabel('input, x')
plt.ylabel('output, y')
plt.show()
