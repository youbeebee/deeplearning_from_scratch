import numpy as np

# 5.5.1 ReLU 계층
"""
y = x (x > 0)
    0 (x <= 0)
∂y/∂x  = 1 (x > 0)
         0 (x <= 0)

ReLU의 계산 그래프
if x > 0
x     → relu → y
∂L/∂y ← relu ← ∂L/∂y

if x <= 0
x → relu → y
0 ← relu ← ∂L/∂y
"""


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


# 5.5.2 Sigmoid 계층
"""
y = 1 / (1 + exp(-x))

시그모이드의 계산 그래프
x → × → exp → + → / → y
-1↗         1↗

1단계
'/'노드
y = 1/x
∂y/∂x = -1/x^2 = -y²
상류에서 흘러온 값에 -y^2(순전파의 출력을 제곱하고 마이너스)을 곱해서 하류로 전달 : -∂L/∂y*y²

2단계
'+'노드
상류의 값을 그대로 하류로 전달 : -∂L/∂y*y²

3단계
'exp'노드
y = exp(x)
∂y/∂x = exp(x)
상류의 값에 순전파 때의 출력(이 경우엔 exp(-x))을 곱해 하류로 전달 : -∂L/∂y*y²*exp(-x)

4단계
'×'노드
순전파 때의 값을 서로 바꿔 곱함(여기서는 * -1) : ∂L/∂y*y²*exp(-x)
∂L/∂y*y^2*exp(-x)는 정리하면 ∂L/∂y*y(1-y)가 된다.(순전파의 출력만으로 계산할 수 있다)

정리
x            → sigmoid → y
∂L/∂y*y(1-y) ← sigmoid ← ∂L/∂y
"""


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


if __name__ == '__main__':
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])
    print(x)
    """
    [[ 1.  -0.5]
     [-2.   3. ]]
    """

    mask = (x <= 0)
    print(mask)
    """
    [[False  True]
     [ True False]]
    """

    # 5.6 Affine/Softmax 계층 구현하기
    # 5.6.1 Affine 계층
    """
    신경망의 순전파에서는 가중치 신호의 총합을 계산하기 때문에 행렬의 내적을 사용했다.(3.3 참고)
    """
    X = np.random.rand(2)     # 입력
    W = np.random.rand(2, 3)  # 가중치
    B = np.random.rand(3)     # 편향

    print(X.shape)  # (2,)
    print(W.shape)  # (2, 3)
    print(B.shape)  # (3,)

    Y = np.dot(X, W) + B
    # 신경망의 순전파 때 수행하는 행렬의 내적은 기하학에서는 어파인 변환이라고 한다.

"""
Affine 계층의 계산 그래프
X, W, B는 행렬(다차원 배열)

1. X ↘    X·W
       dot → + → Y
2. W ↗ 3.B ↗

1. ∂L/∂X = ∂L/∂Y·W^T
   (2,)    (3,)  (3,2)
2. ∂L/∂W = X^T·∂L/∂Y
   (2,3)  (2,1)(1,3)
3. ∂L/∂B = ∂L/∂Y
   (3,)    (3,)
W^T : W의 전치행렬(W가 (2,3)이라면 W^T는(3,2)가 된다.)
X = (x0, x1, x2, ..., xn)
∂L/∂X = (∂L/∂x0, ∂L/∂x1, ∂L/∂x2, ..., ∂L/∂xn)
따라서 X와 ∂L/∂X의 형상은 같다.
"""

# 5.6.2 배치용 Affine 계층
"""
입력 데이터로 X 하나만이 아니라 데이터 N개를 묶어 순전파하는 배치용 계층을 생각

배치용 Affine 계층의 계산 그래프
X의 형상이 (N,2)가 됨.

1. ∂L/∂X = ∂L/∂Y·W^T
   (N,2)   (N,3) (3,2)
2. ∂L/∂W = X^T·∂L/∂Y
   (2,3)  (2,N)(N,3)
3. ∂L/∂B = ∂L/∂Y의 첫 번째 축(0축, 열방향)의 합.
   (3,)    (N,3)

편향을 더할 때에 주의해야 한다. 순전파 때의 편향 덧셈은 X·W에 대한 편향이
각 데이터에 더해진다. 예를 들어 N=2일 경우 편향은 두 데이터 각각에 더해진다.
"""
if __name__ == '__main__':
    X_dot_W = np.array([[0, 0, 0], [10, 10, 10]])
    B = np.array([1, 2, 3])
    print(X_dot_W)
    """
    [[ 0  0  0]
     [10 10 10]]
    """
    print(X_dot_W + B)
    """
    [[ 1  2  3]
     [11 12 13]]
    """
    """
    순전파의 편향 덧셈은 각각의 데이터에 더해지므로
    역전파 때는 각 데이터의 역전파 값이 편향의 원소에 모여야 한다.
    """
    dY = np.array([[1, 2, 3], [4, 5, 6]])
    print(dY)
    """
    [[1 2 3]
     [4 5 6]]
    """
    dB = np.sum(dY, axis=0)
    print(dB)  # [5 7 9]
    """
    데이터가 두 개일 때 편향의 역전파는 두 데이터에 대한 미분을 데이터마다
    더해서 구한다.
    np.sum()에서 0번째 축(데이터를 단위로 한 축. axis=0)에 대해서 합을 구한다.
    """


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


# 5.6.3 Softmax-with-Loss 계층
"""
소프트맥스 계층 : 입력 값을 정규화(출력의 합이 1이 되도록 변경)하여 출력
학습과 추론 중 학습에서 주로 사용
소프트맥스 계층과 손실 함수(교차 엔트로피 오차)를 포함해 계산 그래프를 그림
자세한 역전파 계산은 부록A 참고.

간소화한 Softmax-with-Loss계층의 계산 그래프
a1   →    |         | → y1 → |         |
y1 - t1 ← |         |   t1 ↗  | Cross   |
a2   →    | Softmax | → y2 → | Entropy | → L
y2 - t2 ← |         |   t2 ↗  | Error   | ← 1
a3   →    |         | → y3 → |         |
y3 - t3 ←               t3 ↗
입력 : (a1, a2, a3)
정규화된 출력 : (y1, y2, y3)
정답 레이블 (t1, t2, t3)
손실 : L

역전파로 Softmax 계층의 출력과 정답 레이블의 차분 값
(y1 - t1, y2 - t2, y2 - t2)이 전달됨.
이는 교차 엔트로피 오차 함수가 그렇게 설계되었기 때문.
항등 함수의 손실 함수로는 평균 제곱 오차를 사용하는데,
그럴 경우 역전파의 결과가 (y1 - t1, y2 - t2, y2 - t2)로 말끔히 떨어짐.

ex) 정답 레이블 t = (0, 1, 0) 일 때,
소프트맥스가 (0.3, 0.2, 0.5)를 출력했다고 할 때, 소프트맥스 계층의 역전파는
(0.3, -0.8, 0.5)로 앞 계층에 큰 오차를 전파하게 됨
소프트맥스가 (0.01, 0.99, 0.)을 출력했다면 역전파는 (0.01, -0.01, 0)
으로 오차가 작아짐
"""


# yk = exp(ak) / ∑(i=1 to n)(exp(ai))
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 오버플로 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def cross_entropy_error(y, t):
    delta = 1e-7  # 0일때 -무한대가 되지 않기 위해 작은 값을 더함
    return -np.sum(t * np.log(y + delta))


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 손실
        self.y = None     # softmax의 출력
        self.t = None     # 정답 레이블(원-핫 벡터)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)  # 3.5.2, 4.2.2에서 구현
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = self.y - self.t / batch_size

        return dx


if __name__ == '__main__':
    swl = SoftmaxWithLoss()
    a = np.array([1, 8, 3])   # 비슷하게 맞춤
    t = np.array([0, 1, 0])
    print(swl.forward(a, t))  # 0.0076206166295
    print(swl.backward())     # [ 0.00090496  0.65907491  0.00668679]

    a = np.array([1, 3, 8])   # 오차가 큼
    print(swl.forward(a, t))  # 5.00760576266
    print(swl.backward())   # [  9.04959183e-04 -3.26646539e-01 9.92408247e-01]
