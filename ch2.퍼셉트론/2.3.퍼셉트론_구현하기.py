import numpy as np

# 2.3.1 간단한 구현부터
'''
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
'''

# 2.3.2 가중치와 편향 도입
x = np.array([0, 1])  # 입력
w = np.array([0.5, 0.5])  # 가중치
b = -0.7  # 편향

print(w * x)                # [0. 0.5]
print(np.sum(w * x))        # 0.5
print(np.sum(w * x) + b)    # -0.2


# 2.3.3 가중치와 편향 구현하기
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])  # AND와는 가중치(w, b)만 다르다
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])  # AND와는 가중치(w, b)만 다르다
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


print("AND")
print(AND(0, 0))  # 0
print(AND(0, 1))  # 0
print(AND(1, 0))  # 0
print(AND(1, 1))  # 1

print("NAND")
print(NAND(0, 0))  # 1
print(NAND(0, 1))  # 1
print(NAND(1, 0))  # 1
print(NAND(1, 1))  # 0

print("OR")
print(OR(0, 0))  # 0
print(OR(0, 1))  # 1
print(OR(1, 0))  # 1
print(OR(1, 1))  # 1


# 2.4.1 XOR
'''
XOR게이트는 단층(선형) 퍼셉트론으로는 구현 불가능하다.
 = 단층 퍼셉트론으로는 비선형 영역을 분리할 수 없다.
다층 퍼셉트론multi-layer perceptron을 통해 구현 가능
'''


# 2.5.1 기존 게이트 조합하기
# XOR(x1, x2) = AND(NAND(x1, x2), OR(x1, x2))

# 2.5.2 XOR 게이트 구현하기
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


print("XOR")
print(XOR(0, 0))  # 0
print(XOR(0, 1))  # 1
print(XOR(1, 0))  # 1
print(XOR(1, 1))  # 0

# XOR은 2층 퍼셉트론이다.
# 2층 퍼셉트론(=비선형 시그노이드 함수)를 활성화 함수로 사용하면
# 임의의 함수를 표현할 수 있다는 사실이 증명되어 있다.
