import sys, os
sys.path.append('C:/Users/정용희/Desktop/프로그래밍/python/PRE/deep-learning-from-scratch')
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)   #정규분포로 초기화
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss
net = simpleNet()
print(net.W)
'''
[[random rand rd]
 [ rd rd rd]]
 '''
x = np.array([0.6, 0.9])
p = net.predict(x)
print('가중치:\n', p)
print('확률 최댓값 인덱스->', np.argmax(p))     #최댓값 인덱스

t = np.array([0, 0, 1]) #정답 레이블
print('손실함수 최적값->', net.loss(x, t))   #손실함수 최적값

#일관성 up 버젼
def f(W):
    return net.loss(x, t)
dW = numerical_gradient(f, net.W)
print('일관성 up버젼\n', dW)

#람다 버젼
f = lambda w: net.loss(x, t)
dw = numerical_gradient(f, net.W)
print('람다 버젼\n', dw)