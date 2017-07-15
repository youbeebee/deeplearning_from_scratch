import numpy as np
import sys
import os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

# 4.2.3 미니배치 학습
# 훈련 데이터 전체에 대한 오차함수
# E = -1/N * ∑ _n (∑ _k (tk * log(yk)))
# N : 데이터의 개수
# 훈련 데이터 전체에 대한 손실 함수를 계산하기에는 시간이 오래걸리기 때문에
# 일부를 추려 전체의 근사치로 이용할 수 있다.
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=False)

print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # 원-핫 인코딩 된 정답 레이블 (60000, 10)

# 무작위 10개 추출
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]


# 4.2.4 (배치용) 교차 엔트로피 오차 구현하기
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arange(batch_size), t])) / batch_size


# 4.2.5 왜 손실 함수를 설정하는가?
# 신경망을 학습할 때 정확도를 지표로 삼아서는 안 된다.
# 정확도를 지표로 하면 매개변수의 미분이 대부분의 장소에서 0이 되기 때문이다.
# (매개변수의 미소한 변화에는 거의 반응을 보이지 않고 그 값이 분연속적으로 변화)
