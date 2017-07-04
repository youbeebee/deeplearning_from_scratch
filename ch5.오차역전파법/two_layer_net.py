import sys
import os
import numpy as np
from collections import OrderedDict
sys.path.append(os.pardir)
from common.layers import *
from common.gradient import numerical_gradient

# 5.7.1 신경망 학습의 전체 그림
"""
(4.5와 동일)
전제
신경망에는 적응 가능한 가중치와 편향이 있고, 이 가중치와 편향을 훈련 데이터에 적응하도록 조정하는 과정을 '학습'이라 한다.
신경망 학습은 다음과 같이 4단계로 수행한다.

1단계 - 미니배치
훈련 데이터 중 일부를 무작위로 가져온다. 이렇게 선별한 데이터를 미니배치라 하며,
그 미니배치의 손실함수 값을 줄이는 것이 목표이다.

2단계 - 기울기 산출
미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구한다.
기울기는 손실 함수의 값을 가장 작게 하는 방향을 제시한다.

3단계 - 매개변수 갱신
가중치 매개변수를 기울기 방향으로 아주 조금 갱신한다.

4단계 - 반복
1~3단계를 반복한다.

수치 미분과 오차역전파법은 2단계에서 사용
수치 미분은 구현은 쉽지만 계산이 오래걸림
오차역전파법을 통해 기울기를 효율적이고 빠르게 구할 수 있음
"""

# 5.7.2 오차역전파법을 이용한 신경망 구현하기
"""
TwoLayerNet 클래스로 구현
 * 클래스의 인스턴스 변수
params : 신경망의 매개변수를 보관하는 딕셔너리 변수.
        params['W1']은 1번째 층의 가중치, params['b1']은 1번째 층의 편향.
        params['W2']은 2번째 층의 가중치, params['b2']은 2번째 층의 편향.
layers : 신경망의 계층을 보관하는 순서가 있는 딕셔너리 변수
        layers['Affine1'], layers['Relu1'], layers['Affine2']와 같이
        각 계층을 순서대로 유지
lastLayer : 신경망의 마지막 계층(여기서는 SoftmaxWithLoss)

 * 클래스의 메서드
__init__(...) : 초기화 수행
predict(x) : 예측(추론)을 수행한다. x는 이미지 데이터
loss(x, t) : 손실함수의 값을 구한다. x는 이미지 데이터, t는 정답 레이블
accuracy(x, t) : 정확도를 구한다.
numerical_gradient(x, t) : 가중치 매개변수의 기울기를 수치 미분으로 구함(앞 장과 같음)
gradient(x, t) : 가중치 매개변수의 기울기를 오차역전파법으로 구함
"""


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
        weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = \
            Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = \
            Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        # 순전파
        self.loss(x, t)

        # 역전파
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
