import sys, os
sys.path.append('C:/Users/정용희/Desktop/프로그래밍/python/PRE/deep-learning-from-scratch')
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

#데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize= True, one_hot_label= True)

network = TwoLayerNet(input_size = 784, hidden_size= 50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

print('수치 미분법')
#각 가중치의 차이의 절댓값을 구한 후, 그 절댓값들의 평균을 낸다.
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))
'''
수치 미분으로 구한 기울기
W1:2.382437633784032e-06
b1:2.066807251992833e-05
W2:6.127078778183406e-09
b2:1.4014499095799104e-07
'''

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size/batch_size, 1)
print('오차역전법')
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #오차 역전법으로 기울기 구함
    grad = network.gradient(x_batch, t_batch)

    #갱신하기
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)