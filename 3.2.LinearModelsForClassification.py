import numpy as np
import matplotlib.pyplot as plt
import utils as uu
from scipy.stats import multivariate_normal
import matplotlib.mlab as mlab
import matplotlib

# generate data
# number of samples
n1 = 3
n2 = 3
d = 2  # number of features
np.random.seed(42)
# X1 = np.random.multivariate_normal([-0.5, -0.5], 0.2 * np.eye(d), n1).T
X1 = np.array([[-5, -1, 1], [1, -5, 0]])  # book's example
y1 = -np.ones((1, n1))

# X2 = np.random.multivariate_normal([0.5, 0.5], 0.2 * np.eye(d), n2).T
X2 = np.array([[-0.5, 1, 5], [-0.5, 5, 4]])  # book's example
y2 = np.ones((1, n2))

# weights prior
Sigma_prior = np.eye(d)

# plot
f, axarr = plt.subplots(2, 2)
plt.tight_layout()

# plot weights pdf
w_lim = 4
w_grid_num = 50
w1 = np.linspace(-w_lim, w_lim, w_grid_num)
w2 = np.linspace(-w_lim, w_lim, w_grid_num)
W1, W2 = np.meshgrid(w1, w2)
w_grid = np.empty(W1.shape + (2,))
w_grid[:, :, 0] = W1
w_grid[:, :, 1] = W2

weights_pdf = multivariate_normal([0, 0], Sigma_prior)
axarr[0, 0].contour(W1, W2, weights_pdf.pdf(w_grid), 3, colors='k')
axarr[0, 0].axis('square')
axarr[0, 0].grid(linestyle=':', alpha=0.5, markevery=20)
axarr[0, 0].set_xlabel('$w_1$')
axarr[0, 0].set_ylabel('$w_2$')

# plot training points
axarr[0, 1].scatter(X1[0, :], X1[1, :], marker='x', color='k')
axarr[0, 1].scatter(X2[0, :], X2[1, :], marker='o', facecolors='none', color='k')
axarr[0, 1].set_xlim([-5, 5])
axarr[0, 1].axis('square')
axarr[0, 1].grid(linestyle=':', alpha=0.5, markevery=20)
axarr[0, 1].set_xlabel('$x_1$')
axarr[0, 1].set_ylabel('$x_2$')

# plot weights posterior
axarr[1, 0].contour(W1, W2, uu.weights_post_pdf(w_grid, X1, X2, y1, y2, Sigma_prior), 4, colors='k')
axarr[1, 0].axis('square')
axarr[1, 0].grid(linestyle=':', alpha=0.5, markevery=20)
axarr[1, 0].set_xlabel('$w_1$')
axarr[1, 0].set_ylabel('$w_2$')

# plot predictive distribution
x_lim = 5
x_grid_num = 30
x1 = np.linspace(-x_lim, x_lim, x_grid_num)
x2 = np.linspace(-x_lim, x_lim, x_grid_num)
XX1, XX2 = np.meshgrid(x1, x2)
x_grid = np.empty(XX1.shape + (2,))
x_grid[:, :, 0] = XX1
x_grid[:, :, 1] = XX2

predictive_pdf = np.zeros(XX1.shape)
for i in range(x_grid_num):
    for j in range(x_grid_num):
        x_star = x_grid[i, j, :]
        predictive_pdf[i, j] = uu.lin_reg_predict(x_star, w_grid, X1, X2, y1, y2, Sigma_prior)

axarr[1, 1].scatter(X1[0, :], X1[1, :], marker='x', color='k')
axarr[1, 1].scatter(X2[0, :], X2[1, :], marker='o', facecolors='none', color='k')
cn = axarr[1, 1].contour(XX1, XX2, predictive_pdf, 7, colors='k', linewidths=0.5)

axarr[1, 1].clabel(cn, inline=1, fontsize=7, fmt='%.1f')
axarr[1, 1].axis('square')
axarr[1, 1].grid(linestyle=':', alpha=0.5, markevery=20)
axarr[1, 1].set_xlabel('$x_1$')
axarr[1, 1].set_ylabel('$x_2$')

plt.show()
