import numpy as np

x = np.random.rand(10, 1, 28, 28)
x.shape # (10, 1, 28, 28)

x[0].shape  #(1, 28, 28)
x[1].shape  #(1, 28, 28)

x[0, 0] #또는 x[0][0]