import sys, os
sys.path.append('C:/Users/정용희/Desktop/프로그래밍/python/PRE/deep-learning-from-scratch')
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        #가중치를 초기화한다.
        self.params = {} #신경망 매개변수 저장할 딕셔너리
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        #layer : 순서가 있는 딕셔너리 변수로, 신경망의 계층을 보관
        self.layers = OrderedDict() # 신경망의 계층을 OrderedDict에 보관(순서있음)
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    #추론
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    #x는 입력 데이터, t는 정답 레이블, 손실 함수값을 구한다.
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t) #lastLayer : 신경망의 마지막 계층, 여기서는 SoftmaxWithLoss 층
    
    #정확도를 구한다.
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    #가중치 매개변수의 기울기를 수치 미분 상으로 구한다.
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads
    
    #가중치 매개변수의 기울기를 오차역전파법으로 구한다.
    def gradient(self, x, t):
        #순전파
        self.loss(x, t)

        #역전파
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()#역전파 일때에는 순서를 거꾸로 해야함.
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads