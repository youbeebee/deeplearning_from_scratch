import numpy as np
import matplotlib.pyplot as plt
from simple_convnet import SimpleConvNet

# 7.6.1 1번째 층의 가중치 시각화하기
"""
MNIST 데이터셋의 CNN 학습에서 1번째 층의 합성곱 계층의 가중치는 형상이 (30, 1, 5, 5)였다.
(필터 30개, 채널 1개, 5*5크기) - 1개의 필터는 1채널의 회색조 이미지로 시각화할 수 있다.

학습 전 후의 필터를 이미지로 나타내어 비교한다. 구현은 visualize_filter.py 참고
"""


def filter_show(filters, nx=8, margin=3, scale=10):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


network = SimpleConvNet()
# 무작위(랜덤) 초기화 후의 가중치
filter_show(network.params['W1'])

# 학습된 가중치
network.load_params("params.pkl")
filter_show(network.params['W1'])

"""
학습 전 필터는 무작위로 규칙성이 없지만 학습을 마친 필터는 줄무늬, 덩어리 등 규칙을 띈다.
이러한 필터는 에지(색상이 바뀐 경계), 블롭blob(국소적으로 덩어리진 영역) 등을 인식한다.
Lena.png에 이 필터들을 적용한 결과는 apply_filter.py 참고
"""

# 7.6.2 층 깊이에 따른 추출 정보 변화
"""
1번째 층의 합성곱 계층망에서는 에지나 블롭 등의 저수준 정보가 추출되고 계층이 깊어질수록
추출되는 정보는 더 추상화된다.(에지 -> 텍스처 -> 사물의 일부 등)
"""

# 7.7 대표적인 CNN
"""
LeNet : 1998년에 제안된 손글씨 숫자를 인식하는 네트워크.
합성곱 계층과 풀링 계층(정확히는 원소를 줄이기만 하는 서브샘플링 계층)을 반복하고
마지막으로 완전연결 계층을 거치면서 결과를 출력한다. 활성화 함수로 시그모이드를 사용한다.

AlexNet : 2012년에 발표된 네트워크. 딥러닝 열풍을 일으키는데 큰 역할을 했다.
LeNet과 계층 구조는 크게 바뀌지 않았지만 다음과 같은 변화가 있다.
 * 활성화 함수로 ReLU 사용
 * LRN(Local Response Normalization)이라는 국소적 정규화를 실시하는 계층 사용
 * 드롭아웃을 사용

두 모델의 네트워크 구성에는 큰 차이가 없으나 빅 데이터, GPU연산 등으로 인해 큰 발전이 있었다.
"""
