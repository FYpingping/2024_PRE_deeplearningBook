import numpy as np

def sum_squares_error(y, t):
    return 0.5*np.sum((y-t)**2)

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]    #소프트 맥스 함수 출력
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]      #one hot encoding

print(sum_squares_error(np.array(y), np.array(t)))
#0.097500000... 이 나옴

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(sum_squares_error(np.array(y), np.array(t)))
#0.5975 출력

'''첫번째의 예에서 오차 제곱합이 더 작으므로 첫번째의 예시가 더 정답에 가깝다고 판단'''