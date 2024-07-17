import sys, os
sys.path.append('C:/Users/정용희/Desktop/프로그래밍/python/PRE/deep-learning-from-scratch')
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD
import numpy as np

#데이터 6만개 가져오기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
#오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train[:300]
t_train = t_train[:300]
#784개의 입력층, 6개의 히든층, 출력층 10개
network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10)
optimizer = SGD(lr = 0.01) #학습률 0.01인 SGD로 매개변수 갱신
max_epochs = 201    #최대 에폭스 201
train_size = x_train.shape[0]   #훈련 개수
batch_size = 100    #묶음 개수

train_loss_list = []
train_acc_list = []
test_acc_list = []
#에폭당 반복 횟수
iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    #train_size 데이터셋에서 batch_size 크기의 무작위 샘플 인덱스를 선택하여 미니배치
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #기울기 구하기
    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break