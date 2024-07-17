import sys, os
sys.path.append('C:/Users/정용희/Desktop/프로그래밍/python/PRE/deep-learning-from-scratch')
from common.util import im2col
import numpy as np

# im2col(input_data, filter_h, filter_w, pad = 0)

x1 = np.random.rand(1, 3, 7, 7) #(데이터 수, 채널 수, 높이, 너비)
col1 = im2col(x1, 5,5, stride = 1, pad = 0)
print(col1.shape)   #(9, 75)

x2 = np.random.rand(10, 3, 7, 7)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)   #(90,75)