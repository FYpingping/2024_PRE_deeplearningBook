import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    # __init__ : 객체 초기화
    #input_size: 입력층의 뉴런 수
    #hidden_size: 은닉층의 뉴런 수
    #output_size: 출력층의 뉴런 수
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #params : 신경망의 매개변수를 보관하는 딕셔너리
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    #예측(추론) 수행
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    
    #손실 함수의 값을 구한다. x: 이미지 데이터, t: 정답 레이블
    def loss(self, x, t):
        y = self.predictI(x)

        return cross_entropy_error(y, t)
    
    #정확도를 구함
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t)/float(x.shape[0])
        return accuracy
    
    #가중치 매개변수의 기울기를 구함
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        #grads: 기울기를 보관하는 딕셔너리
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    
    #numerical_gradient() 성능 개선 메소드
    def gradient(self, x, t):
        {
            #5장에서...
        }
    