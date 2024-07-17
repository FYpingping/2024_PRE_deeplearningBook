import sys, os
sys.path.append('C:/Users/정용희/Desktop/프로그래밍/python/PRE/deep-learning-from-scratch')
from common.functions import softmax, cross_entropy_error

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None    #손실
        self.y = None       #softmax 출력
        self.t = None       #정답 레이블(원-핫 벡터)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t)/batch_size
        return dx