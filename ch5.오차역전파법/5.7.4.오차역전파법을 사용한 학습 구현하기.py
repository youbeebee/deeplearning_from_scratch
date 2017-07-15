import sys
import os
import numpy as np
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼 파라메터
iters_num = 10000  # 반복횟수
train_size = x_train.shape[0]
batch_size = 100  # 미니배치 크기
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # print(i)
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 오차역전파법으로 기울기 계산
    grad = network.gradient(x_batch, t_batch)

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
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

"""
train acc, test acc | 0.0992833333333, 0.1032
train acc, test acc | 0.898, 0.9026
train acc, test acc | 0.92135, 0.9216
train acc, test acc | 0.936016666667, 0.9337
train acc, test acc | 0.945316666667, 0.9431
train acc, test acc | 0.94675, 0.9427
train acc, test acc | 0.954766666667, 0.9521
train acc, test acc | 0.9602, 0.9551
train acc, test acc | 0.9634, 0.9581
train acc, test acc | 0.9656, 0.9597
train acc, test acc | 0.9683, 0.9615
train acc, test acc | 0.970516666667, 0.9629
train acc, test acc | 0.97305, 0.9649
train acc, test acc | 0.9731, 0.9661
train acc, test acc | 0.975916666667, 0.9659
train acc, test acc | 0.976383333333, 0.9666
train acc, test acc | 0.977916666667, 0.969
[Finished in 45.5s]
"""

# 5.8 정리
"""
계산 과정을 시각적으로 보여주는 방법인 계산 그래프에 대해 학습했다.
계산 그래프를 통해 신경망의 동작과 오차역전파법을 설명하고, 계층이라는 단위로 구현했다.

"""
