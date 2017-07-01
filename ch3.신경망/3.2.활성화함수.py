import numpy as np
import matplotlib.pylab as plt

# 3.2.1 시그모이드 함수
# h(x) = 1/(1 + exp(-x))


# 3.2.2 계단 함수 구현하기
# 계단 함수 : 입력값을 경계로 출력이 바뀌는 함수
'''
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
'''


# numpy 배열을 위한 구현
def step_function(x):
    # y = x > 0
    # return y.astype(np.int)
    return np.array(x > 0, dtype=np.int)


# 3.2.3 계단 함수의 그래프
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # y축의 범위 지정
# plt.show()


# 3.2.4 시그모이드 함수 구현하기
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # y축의 범위 지정
plt.show()

# 3.2.6 비선형 함수
# 선형 함수는 은닉층이 없는 네트워크로 표현할 수 있다.
# 층을 쌓는 혜택을 얻고 싶다면 활성화 함수로는 비선형 함수를 사용해야 한다.


# 3.2.7 ReLU 함수
# Rectified Linear Unit - 입력이 0을 넘으면 입력 그대로, 아니면 0을 출력하는 함수
def relu(x):
    return np.maximum(0, x)
