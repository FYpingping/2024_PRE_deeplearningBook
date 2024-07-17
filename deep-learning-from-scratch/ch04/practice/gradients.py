import numpy as np
from diff import *

def numerical_gradient(f, x):
    h = 1e-4    #0.0001
    grad = np.zeros_like(x)     #x와 형상이 같은 배열을 상징

    for idx in range(x.size):
        tmp_val = x[idx]

        #f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)
        #f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    
    return grad

'''(3,4), (0,2), (3,0)의 기울기 구하기'''
print(numerical_gradient(function_2, np.array([3.0, 4.0])))     #[6. 8.]
print(numerical_gradient(function_2, np.array([0.0, 2.0])))     #[0. 4.]
print(numerical_gradient(function_2, np.array([3.0, 0.0])))     #[6. 0.]

#f는 최적화 하려는 함수, init_x는 초기값, lr은 learning rate를 의미하는 학습률, step_num은 경사법에 따른 반복횟수
def gradient_descent(f, init_x, lr= 0.01, step_num = 100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)     #기울기 구하기
        x -= lr * grad      #기울기 * 학습률
    
    return x

'''경사법으로 f(x0, x1) = x0^2 + x1^2의 최솟값을 구하라'''
init_x = np.array([-3.0, 4.0])
print('학습률: 0.1 ->', gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))
#[-6.11110793e-10  8.14814391e-10]

#학습률이 너무 크면 발산해버림
init_x = np.array([-3.0, 4.0])
print('학습률: 10.0 ->', gradient_descent(function_2, init_x=init_x, lr = 10.0, step_num=100))
#[-2.58983747e+13 -1.29524862e+12]

#학습률이 너무 작으면 갱신이 거의 안됨
init_x = np.array([-3.0, 4.0])
print('학습률: 1e-10 ->', gradient_descent(function_2, init_x=init_x, lr = 1e-10, step_num=100))
#[-2.99999994  3.99999992]