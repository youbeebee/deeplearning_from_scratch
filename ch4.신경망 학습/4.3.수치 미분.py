import numpy as np
import matplotlib.pylab as plt


# 4.3.1 미분
# 나쁜 구현 예
def numerical_diff_bad(f, x):
    h = 10e-50
    return (f(x + h) - f(x)) / h
# h값이 너무 작아 반올림 오차를 일으킬 수 있음 10e-4정도가 적당하다고 알려짐
# 전방 차분에서는 차분이 0이 될 수 없어 오차가 발생
#  -> 오차를 줄이기 위해 중심 차분을 사용


def numerical_diff(f, x):
    h = 10e-4
    return (f(x + h) - f(x - h)) / (2 * h)


# 4.3.2 수치 미분의 예
# y = 0.01x² + 0.1x
def function_1(x):
    return 0.01*x**2 + 0.1*x


x = np.arange(0.0, 20.0, 0.1)  # 0에서 20까지 간격 0.1인 배열 x를 만든다.
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
# plt.show()

# x = 5, 10일때 미분
print(numerical_diff(function_1, 5))   # 0.200000000000089
print(numerical_diff(function_1, 10))  # 0.29999999999996696


# 접선의 함수를 구하는 함수
def tangent_line(f, x):
        d = numerical_diff(f, x)
        # print(d)
        y = f(x) - d*x
        return lambda t: d*t + y


tf = tangent_line(function_1, 5)
y2 = tf(x)
plt.plot(x, y2)
plt.show()


# 4.3.3 편미분
# f(x0, x1) = x0² + x1²
def function_2(x):
    return x[0]**2 + x[1]**2
    # or return np.sum(x**2)


# x0 = 3, x1 = 4일 때, x0에 대한 편미분을 구하라.
def function_tmp1(x0):
    return x0**2 + 4.0**2.0


# x0 = 3, x1 = 4일 때, x1에 대한 편미분을 구하라.
def function_tmp2(x1):
    return 3.0**2.0 + x1 * x1


print(numerical_diff(function_tmp1, 3.0))  # 5.999999999998451
print(numerical_diff(function_tmp2, 4.0))  # 8.000000000000895
