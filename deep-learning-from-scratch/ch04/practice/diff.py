import numpy as np


#미분 나쁜 구현의 예시
def numerical_diff(f, x):
    h = 1e-50
    return (f(x+h)-f(x))/h
#1e-50 같이 작은 숫자는 반올림 오차에 의해 생략이 되버림.


print(np.float32(1e-50))
# 0.0 출력

def numerical_central_diff(f, x):
    h = 1e-4    #0.0001
    return (f(x+h)-f(x-h))/(2*h)

def function_1(x):
    return 0.01*x**2+0.1*x

print(numerical_central_diff(function_1, 5))    #0.1999999999990898
print(numerical_central_diff(function_1, 10))   #0.2999999999986347

#편미분
#f(x0, x1) = x0^2 + x1^2
def function_2(x):
    return x[0]**2 + x[1]**2

'''x0 = 3, x1 = 4일때, x0에 대한 편미분 df/dx0 을 구하라'''
# x1부분은 어차피 상수로 취급하므로
def function_tmp1(x0):
    return x0*x0 + 4.0**2.0
print(numerical_central_diff(function_tmp1, 3.0))   #6.00...00378

'''x0 = 3, x1 = 4일때, x1에 대한 편미분 df/dx1 을 구하라'''
# x0부분은 어차피 상ㅇ수로 취급하므로
def function_tmp2(x1):
    return 3.0**2.0 + x1*x1

print(numerical_central_diff(function_tmp2, 4.0))   #7.99...99119