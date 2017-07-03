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
∂y/∂x = -1/x^2 = -y^2
상류에서 흘러온 값에 -y^2(순전파의 출력을 제곱하고 마이너스)을 곱해서 하류로 전달 : -∂L/∂y*y^2

2단계
'+'노드
상류의 값을 그대로 하류로 전달 : -∂L/∂y*y^2

3단계
'exp'노드
y = exp(x)
∂y/∂x = exp(x)
상류의 값에 순전파 때의 출력(이 경우엔 exp(-x))을 곱해 하류로 전달 : -∂L/∂y*y^2*exp(-x)

4단계
'×'노드
순전파 때의 값을 서로 바꿔 곱함(여기서는 * -1) : ∂L/∂y*y^2*exp(-x)
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
