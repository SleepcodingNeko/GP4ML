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
X1 = np.random.multivariate_normal([-0.5, -0.5], 0.2 * np.eye(d), n1).T
X1 = np.array([[-5, -1, 1], [1, -5, 0]])  # book's example
y1 = np.ones((1, n1))

X2 = np.random.multivariate_normal([0.5, 0.5], 0.2 * np.eye(d), n2).T
X2 = np.array([[-0.5, 1, 5], [-0.5, 5, 4]])  # book's example
y2 = -np.ones((1, n2))

# weights prior
Sigma_prior = np.eye(d)

# plot
f, axarr = plt.subplots(2, 2)
plt.tight_layout()

# plot weights pdf
w1 = np.linspace(-2, 2, 100)
w2 = np.linspace(-2, 2, 100)
W1, W2 = np.meshgrid(w1, w2)
grid = np.empty(W1.shape + (2,))
grid[:, :, 0] = W1
grid[:, :, 1] = W2
weights_pdf = multivariate_normal([0, 0], Sigma_prior)
axarr[0, 0].contour(W1, W2, weights_pdf.pdf(grid), 3, colors='k')
axarr[0, 0].axis('square')
axarr[0, 0].grid(linestyle=':', alpha=0.5, markevery=20)
axarr[0, 0].set_xlabel('$w_1$')
axarr[0, 0].set_ylabel('$w_2$')

# plot training points
axarr[0, 1].scatter(X1[0, :], X1[1, :], marker='x', color='k')
axarr[0, 1].scatter(X2[0, :], X2[1, :], marker='o', color='k')
axarr[0, 1].set_xlim([-5, 5])
axarr[0, 1].axis('square')
axarr[0, 1].grid(linestyle=':', alpha=0.5, markevery=20)
axarr[0, 1].set_xlabel('$x_1$')
axarr[0, 1].set_ylabel('$x_2$')

# plot weights posterior
bb = 4
w1 = np.linspace(-bb, bb, 200)
w2 = np.linspace(-bb, bb, 200)
W1, W2 = np.meshgrid(w1, w2)
grid = np.empty(W1.shape + (2,))
grid[:, :, 0] = W1
grid[:, :, 1] = W2
axarr[1, 0].contour(W1, W2, uu.weights_post_pdf(grid, X1, X2, y1, y2, Sigma_prior), 4, colors='k')
axarr[1, 0].axis('square')
axarr[1, 0].grid(linestyle=':', alpha=0.5, markevery=20)
axarr[1, 0].set_xlabel('$w_1$')
axarr[1, 0].set_ylabel('$w_2$')

plt.show()
