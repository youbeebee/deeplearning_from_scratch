import numpy as np
import sys
import os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet


# 4.5.2 미니배치 학습 구현하기
# * 주의 : 아주 오래 걸림 *
"""
60000개의 훈련 데이터에서 임의로 100개의 데이터(이미지&정답 레이블)을 추려냄.
100개의 미니배치를 대상으로 확률적 경사 하강법을 수행해 매개변수를 갱신한다.
경사법에 의한 갱신 횟수를 1000번으로 설정하고 갱신할 때마다 손실 함수를 계산한다.
"""
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=False)

# 하이퍼 파라메터
iters_num = 1000  # 반복횟수
train_size = x_train.shape[0]
batch_size = 100  # 미니배치 크기
learning_rate = 0.1
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    print(i)
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch)
    # grad = network.gradient(x_batch, t_batch)  # 다음 장에서 구현할 더 빠른 방법!

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1에폭 당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | "
              + str(train_acc) + ", " + str(test_acc))

# print(train_loss_list)


# 4.5.3 시험 데이터로 평가하기
"""
위의 계산에서 손실 함수의 값이 점점 감소하게 되는데, 이때의 손실 함수는
훈련 데이터의 미니배치에 대한 손실 함수를 말한다.
훈련 데이터 외의 데이터를 올바르게 인식하는지(오버피팅이 일어나지 않았는지) 확인 필요.
1 에폭별로 훈련 데이터와 시험 데이터에 대한 정확도를 기록하도록 수정.
에폭epoch : 학습에서 훈련 데이터를 모두 소진했을 때의 횟수.
10000개를 100개의 미니배치로 학슬할 경우 100회가 1에폭이 된다.

훈련 데이터와 시험 데이터의 정확도 추이가 비슷하다면 오버피팅이 일어나지 않은 것이다.
오버피팅이 발생했다면, 어느 순간부터 시험 데이터에 대한 정확도가 떨어지기 시작한다.
오버피팅이 발생하기 전에 학습을 중단해 오버피팅을 예방하는 기법을 조기 종료early stopping라고 한다.
"""
