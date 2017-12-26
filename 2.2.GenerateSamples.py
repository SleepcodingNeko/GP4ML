import numpy as np
import matplotlib.pyplot as plt
import utils as uu


n = 200
mu = np.zeros(n)
n_samples = 3

x = np.linspace(-10, 10, n)
cov = uu.se_matrix(x)

samples1 = np.random.multivariate_normal(mu, cov, n_samples)

# plot samples
for i in range(n_samples):
    plt.plot(x, samples1[i, :], marker='')
plt.grid(alpha=0.5)
plt.xlabel('x, input')
plt.ylabel('y, output')
plt.show()
