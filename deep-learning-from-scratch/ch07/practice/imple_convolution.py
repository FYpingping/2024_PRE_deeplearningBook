import sys, os
sys.path.append('C:/Users/정용희/Desktop/프로그래밍/python/PRE/deep-learning-from-scratch')
from common.util import im2col
import numpy as np

class Convolution:
    def __init__(self, W, b, stride = 1, pad = 0):  #필터(가중치), 편향, 스트라이드, 패딩
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
    
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T    #필터 전개
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out