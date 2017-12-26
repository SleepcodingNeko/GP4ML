import numpy as np
import matplotlib.pyplot as plt
import utils as uu
from scipy.stats import multivariate_normal
import matplotlib.mlab as mlab
import matplotlib

# generate data
n = 5  # number of samples
d = 2  # number of features
np.random.seed(42)
X = np.random.random((d, n)) * 10 - 5
X[1, :] = 1
w = np.array([[1], [-1]])
f_x = np.dot(w.T, X)
noise = np.random.normal(0, 0.3, n)
y = f_x + noise.T

# likelihood
sigma_n = 1
ML_w = np.dot(np.dot(np.linalg.inv(np.dot(X, X.T)), X), y.T)

# prior
Sigma_prior = 1.0 * np.eye(2)

# posterior
A = (sigma_n**(-2)) * np.dot(X, X.T) + np.linalg.inv(Sigma_prior)
Ainv = np.linalg.inv(A)
mean_post = sigma_n**(-2) * np.dot(np.dot(Ainv, X), y.T)
Sigma_post = Ainv

# prediction
n_test = 20
test_values = np.linspace(-5, 5, n_test)
y_pred = np.zeros(n_test)
y_upper = np.zeros(n_test)
y_lower = np.zeros(n_test)
for i in range(n_test):
    x_star = np.array([test_values[i]])
    x_star = np.concatenate((x_star, np.array([1])), axis=0)
    mean_pred, Sigma_pred = uu.predict(x_star, sigma_n, A, X, y)
    y_pred[i] = mean_pred
    y_upper[i] = mean_pred + Sigma_pred
    y_lower[i] = mean_pred - Sigma_pred


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
axarr[0, 0].set_xlabel('slope, $w_1$')
axarr[0, 0].set_ylabel('intercept, $w_2$')

# plot training points
axarr[0, 1].scatter(X[0, :], y, marker='x', color='k')
axarr[0, 1].set_xlim([-5, 5])
axarr[0, 1].axis('square')
axarr[0, 1].grid(linestyle=':', alpha=0.5, markevery=20)
axarr[0, 1].set_xlabel('input, x')
axarr[0, 1].set_ylabel('output, y')

# plot prior
posterior_pdf = multivariate_normal(ML_w.T[0], 0.1)
axarr[1, 0].contour(W1, W2, posterior_pdf.pdf(grid), 3, colors='k')
axarr[1, 0].axis('square')
axarr[1, 0].grid(linestyle=':', alpha=0.5, markevery=20)
axarr[1, 0].set_xlabel('slope, $w_1$')
axarr[1, 0].set_ylabel('intercept, $w_2$')

# plot posterior
posterior_pdf = multivariate_normal(mean_post.T[0], Sigma_post)
axarr[1, 1].contour(W1, W2, posterior_pdf.pdf(grid), 3, colors='k')
axarr[1, 1].axis('square')
axarr[1, 1].grid(linestyle=':', alpha=0.5, markevery=20)
axarr[1, 1].set_xlabel('slope, $w_1$')
axarr[1, 1].set_ylabel('intercept, $w_2$')

# plot predictive pdf
axarr[0, 1].plot(test_values, y_pred, c='k')
axarr[0, 1].plot(test_values, y_upper, c='k', linestyle=':')
axarr[0, 1].plot(test_values, y_lower, c='k', linestyle=':')
plt.show()
