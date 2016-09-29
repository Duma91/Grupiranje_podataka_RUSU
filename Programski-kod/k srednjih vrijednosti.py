# -*- coding: utf-8 -*-

"""
@author: Andrija
"""

import numpy as np
import pylab as plt

mean = [10, 25]
cov = [[1, 0], [0, 2]]    
mean2 = [0, 0]
cov2 = [[1, 0], [0, 1]]
mean3 = [-50, 20]
cov3 = [[1, 0], [0, 4]]

K = 3   
maxIters = 10

data1 = np.random.multivariate_normal(mean, cov, 200)
data2 = np.random.multivariate_normal(mean2, cov2, 500)
data3 = np.random.multivariate_normal(mean3, cov3, 200)
X = np.vstack((data1, np.vstack((data2, data3))))
np.random.shuffle(X)

centroids = X[np.random.choice(np.arange(len(X)), K), :]
for i in range(maxIters):
    C = np.array([np.argmin([np.dot(x_i - y_k, x_i - y_k) for y_k in centroids]) for x_i in X])
centroids = [X[C == k].mean(axis=0) for k in range(K)]
centroids = np.array(centroids)

plt.ion()
plt.cla()
plt.plot(X[C == 0, 0], X[C == 0, 1], '*b',
X[C == 1, 0], X[C == 1, 1], '*r',
X[C == 2, 0], X[C == 2, 1], '*g')
plt.draw()
plt.ioff()
plt.show()
