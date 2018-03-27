from pyflann import *
import numpy as np

dataset = np.array(
    [[1., 1, 1, 2, 3],
     [1,1,1,1,1],
     [10, 10, 10, 3, 2],
     [100, 100, 2, 30, 1]
     ])
testset = np.array(
    [1., 1, 1, 1, 1])
flann = FLANN()
result, dists = flann.nn(dataset, testset, 3, algorithm="kmeans", branching=32, iterations=7, checks=16)
print(result)