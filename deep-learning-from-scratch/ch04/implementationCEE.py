import sys, os
sys.path.append('C:/Users/정용희/Desktop/프로그래밍/python/PRE/deep-learning-from-scratch')
import numpy as np
from dataset.mnist import load_mnist

def cross_entropy_error(y, t, one_hot_encoding = True):
    if y.ndim ==1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        batch_size = y.shape[0] # y.shape = (y.size, ) 즉, batch_size는 신경망 출력 y의 개수
    if one_hot_encoding == True:
        return print(-np.sum(t*np.log(y+1e-7))/batch_size)     #one_hot_encoding 되었을 경우
    else:
        return print(-np.sum(np.log(y[np.arange(batch_size), t]+1e-7))/batch_size)     #one_hot_encoding 안됐을 경우