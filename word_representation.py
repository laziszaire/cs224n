#### SVD

import numpy as np
from numpy.linalg import svd
# window based co-occurence matrix
X = np.abs(np.floor(np.random.randn(10, 10)))  # vocabulary 共10个词（行），列为每个词在其window中出现的次数
U, S, VH = svd(X)
svd_vector = U[:, S.cumsum()/S.sum() <= .9]


#### iteration based methods
# CBOW
# skip-gram
# negative sampling
# todo hierarchical softmax

