'''신경망 출력의 값이 정답에 가까울수록 오차가 작아지고, 정답에 멀어질수록 오차가 커짐'''

import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7    #아주 작은 값
    return -np.sum(t*np.log(y+delta)) # np.log()에 0을 입력하면 -inf가 되므로 그것을 방지하기 위한 delta

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

print(cross_entropy_error(np.array(y), np.array(t)))
#0.510825457099338 출력

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))
#2.302584092994546 출력