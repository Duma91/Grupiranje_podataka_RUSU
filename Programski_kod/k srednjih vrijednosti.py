"""
@author: Andrija
"""

import numpy as np
import pylab as plt
 
mean = [0, 0]
cov = [[300, 10], [500, 10]]
mean2 = [1, 1]
cov2 = [[100, 30], [20, 40]]
mean3 = [2, 4]
cov3 = [[30, 35], [5, 50]]
K = 3
maxIters = 1
 
data1 = np.random.multivariate_normal(mean, cov, 100)
data2 = np.random.multivariate_normal(mean2, cov2, 50)
data3 = np.random.multivariate_normal(mean2, cov3, 5000)
X = np.vstack((data1, np.vstack((data2, data3))))
np.random.shuffle(X)

centroids = X[np.random.choice(np.arange(len(X)), K), :]
for i in range(maxIters):
    C = np.array([np.argmin([np.dot(x_i - y_k, x_i - y_k) for y_k in centroids]) for x_i in X])
    
plt.ion()
plt.cla()
plt.plot(X[C == 0, 0], X[C == 0, 1], '*b',
         X[C == 1, 0], X[C == 1, 1], '*r',
         X[C == 2, 0], X[C == 2, 1], '*g')
plt.draw()
plt.ioff()
plt.show()