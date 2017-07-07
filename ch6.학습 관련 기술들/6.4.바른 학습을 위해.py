# coding: utf-8
import os
import sys
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD
from common.util import shuffle_dataset

# 6.4.1 오버피팅
"""
오버피팅은 주로 다음의 경우에 일어난다.
 * 매개변수가 많고 표현력이 높은 모델
 * 훈련 데이터가 적음

강제로 오버피팅을 만들기 위해 MNIST 데이터 셋 중 300개만 사용하고 7층 네트워크를 사용해
복잡성을 높인다. 각 층의 뉴런은 100개, 활성화 함수는 ReLU.
"""


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train[:300]
t_train = t_train[:300]

# weight decay（가중치 감쇠） 설정 =======================
weight_decay_lambda = 0  # weight decay를 사용하지 않을 경우
# weight_decay_lambda = 0.1
# ====================================================

network = MultiLayerNet(input_size=784,
                        hidden_size_list=[100, 100, 100, 100, 100, 100],
                        output_size=10,
                        weight_decay_lambda=weight_decay_lambda)
optimizer = SGD(lr=0.01)  # 학습률이 0.01인 SGD로 매개변수 갱신

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc)
              + ", test acc:" + str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break


# 그래프 그리기==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

"""
훈련 데이터의 정확도는 100%지만 시험 데이터는 76% 수준이다.
이는 훈련 데이터에만 적응했기 때문에 훈련 때 사용하지 않은 범용 데이터에는 대응하지 못하는 것이다.
"""

# 6.4.2 가중치 감소
"""
가중치 감소weight decay : 학습 과정에서 큰 가중치에 대해서는 그에 상응하는 큰 패널티를
부과하여 오버피팅을 억제하는 방법

신경망 학습의 목적은 손실 함수의 값을 줄이는 것. 이때 예를 들어 가중치의 제곱 노름(L2 norm)을
손실 함수에 더하면 가중치가 커지는 것을 억제할 수 있다.
W : 가중치
L2 노름에 따른 가중치 감소 = 1/2 * λ * W²
λ(람다) : 정규화의 세기를 조절하는 하이퍼파라미터. 크게 설정할수록 큰 가중치에 대한 패널티가 커짐
이 코드에서는 0.1로 적용함.
결과는 훈련 데이터와 시험 데이터의 정확도 차이가 줄어들고 훈련 데이터의 정확도도 100%에
도달하지 못했음.

NOTE : L2 노름은 각 원소의 제곱들을 더한 것에 해당한다.
가중치 W = (w1, w2, ..., wn)이 있다면, L2 노름은 √(w1² + ... + wn²)이다.
이외에 L1, L∞도 있다.
L1 노름 : 절댓값의 합. |w1| + ... + |wn|
L∞ 노름 : Max 노름. 각 원소의 절댓값 중 가장 큰 것
"""

# 6.4.3 드롭아웃
"""
가중치 감소는 간단하게 구현할 수 있고 어느정도 오버피팅을 방지할 수 있지만 신경망 모델이
복잡해지면 가중치 감소만으로는 대응하기 어려워진다. 이때 드롭아웃 기법을 사용한다.

드롭아웃 : 뉴런을 임의로 삭제하면서 학습하는 방법. 훈련 때 은닉층의 뉴런을 무작위로 골라 삭제한다.
훈련때는 데이터를 흘릴 때마다 삭제할 뉴런을 무작위로 선택하고 시험 때는 모든 뉴런에 신호를 전달.
단, 시험 때는 각 뉴런의 출력에 훈련 때 삭제한 비율을 곱하여 출력한다.(안해도 됨)
"""


class Dropout:
    """
    순전파 때마다 mask에 삭제할 뉴런을 False로 표시한다. mask는 x와 같은 형상의 무작위 배열을
    생성하고 그 값이 dropout_ratio보다 큰 원소만 True로 설정한다.
    역전파 때의 동작은 ReLU와 같다.
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


"""
overfit_dropout.py 참고
훈련과 시험 데이터에 대한 정확도 차이가 줄어듬
훈련 데이터에 대한 정확도가 100%에 도달하지 않음.
epoch:301, train acc:0.73, test acc:0.6315

NOTE : 앙상블 학습ensemble learning : 개별적으로 학습시킨 여러 모델의 출력을 평균내 추론.
앙상블 학습을 사용하면 신경망의 정확도가 몇% 정도 개선된다는 것이 실험적으로 알려져 있음.

앙상블 학습은 드롭아웃과 밀접하다. 학습 때 뉴런을 무작위로 학습하는 것이 매번 다른 모델을
학습시키는 것으로 해석할 수 있다. 추론 때 삭제한 비율을 곱하는 것은 앙상블에서 모델의 평균과 같다.
"""

# 6.5 적절한 하이퍼파라미터 값 찾기
# 6.5.1 검증 데이터
"""
데이터셋을 훈련 데이터와 시험 데이터로 분리해 이용해서 오버피팅과 범용 성능 등을 평가했다.
하이퍼파라미터를 설정하고 검증할 때는 시험 데이터를 사용해서는 안 된다.

시험 데이터를 사용하여 하이퍼파라미터를 조정하면 하이퍼파라미터 값이 시험 데이터에 오버피팅된다.
따라서 하이퍼파라미터를 조정할때는 전용 확인 데이터가 필요하다.
이를 검증 데이터validation data라고 부른다.

NOTE :
 * 훈련 데이터 : 매개변수(가중치와 편향)의 학습에 이용
 * 검증 데이터 : 하이퍼파라미터의 성능을 평가
 * 시험 데이터 : 범용 성능을 확인하기 위해 마지막에(이상적으로는 한 번만) 이용

MNIST는 검증 데이터가 따로 없다. 훈련 데이터에서 20% 정도를 분리해서 사용할 수 있다.
"""

(x_train, t_train), (x_test, t_test) = load_mnist()

# 훈련 데이터를 뒤섞는다.
x_train, t_train = shuffle_dataset(x_train, t_train)

# 20%를 검증 데이터로 분할
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)

x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]


# 6.5.2 하이퍼파라미터 최적화
"""
최적화의 핵심은 하이퍼파라미터의 '최적값'이 존재하는 범위를 조금씩 줄여간다는 것.
대략적인 범위를 설정하고 그 범위에서 무작위로 값을 샘플링 후 그 값으로 정확도를 평가한다.

NOTE : 하이퍼파라미터 최적화에서는 그리드 서치grid search 같은 규칙적인 탐색보다는
무작위 샘플링 탐색이 좋은 결과를 낸다고 알려져 있다.
최종 정확도에 미치는 영향이 하이퍼파라미터마다 다르기 때문이다.

하이퍼 파라미터의 범위는 '대략적으로' 지정한다.(로그 스케일)
딥러닝 학습은 오랜 시간이 걸리기 때문에 나쁠 값은 일찍 포기하는 것이 좋다.
에폭을 작게 하여 1회 평가에 걸리는 시간을 단축하는 것이 효과적이다.

0단계
하이퍼파라미터 값의 범위를 설정한다.
1단계
설정된 범위에서 하이퍼파라미터 값을 무작위로 추출한다.
2단계
1단계에서 샘플링한 값을 사용하여 학습하고, 검증 데이터로 평가한다.(단, 에폭은 작게 설정한다.)
3단계
1~2단계를 특정 횟수(100회 등) 반복하여 정확도의 결과를 보고 하이퍼파라미터의 범위를 좁힌다.

NOTE : 해당 최적와 기법은 수행자의 직관에 많이 의존한다.
더 세련된 기법으로는 베이즈 최적화Bayesian optimization가 있다.
베이즈 정리Bayes' theorem를 이용하여 엄밀하고 효율적으로 최적화를 수행한다.
"""

# 6.5.3 하이퍼파라미터 최적화 구현하기
"""
실제 MNIST에서 최적화를 수행한다.
전체 코드는 pyperparameter_optimization.py 참고

여기서는 가중치 감소 계수를 10^-8 ~ 10^-4
학습률을 10^-6 ~ 10^-2 범위부터 시작
"""
weight_decay = 10**np.random.uniform(-8, -4)
lr = 10 ** np.random.uniform(-6, -2)

"""
=========== Hyper-Parameter Optimization Result ===========
Best-1(val acc:0.77) | lr:0.00642956548737644, weight decay:3.9335005750240353e-05
Best-2(val acc:0.76) | lr:0.009800708251553235, weight decay:4.32104341501499e-05
Best-3(val acc:0.74) | lr:0.008080563079160151, weight decay:4.032225845552401e-07
Best-4(val acc:0.74) | lr:0.008658154225122113, weight decay:1.6387860601920888e-08
Best-5(val acc:0.73) | lr:0.007174090437865117, weight decay:3.3679931489953985e-05
Best-6(val acc:0.71) | lr:0.008092666335553451, weight decay:4.4829857468371013e-05
Best-7(val acc:0.71) | lr:0.006794359177721846, weight decay:2.4009676785451696e-05
Best-8(val acc:0.64) | lr:0.004170771494554204, weight decay:1.0523468836739202e-05
Best-9(val acc:0.63) | lr:0.004110067388120817, weight decay:8.247100494561012e-08
Best-10(val acc:0.61) | lr:0.004359577847920402, weight decay:5.0378978717245236e-08
Best-11(val acc:0.6) | lr:0.0049840874510498935, weight decay:1.0148916836738086e-07
Best-12(val acc:0.57) | lr:0.0032456099306195276, weight decay:2.004358692573245e-05
Best-13(val acc:0.47) | lr:0.0030638198595392277, weight decay:4.298293138860393e-06
Best-14(val acc:0.45) | lr:0.0015322633605043612, weight decay:1.0328552587719503e-05
Best-15(val acc:0.39) | lr:0.0032479013395437556, weight decay:1.2139863191025922e-06
Best-16(val acc:0.36) | lr:0.0015431291922394226, weight decay:1.87520342938985e-05
Best-17(val acc:0.3) | lr:0.001928386461192975, weight decay:7.057939489322395e-07
Best-18(val acc:0.27) | lr:0.001061395877582258, weight decay:2.448175558036332e-07
Best-19(val acc:0.25) | lr:0.00043145521842672074, weight decay:5.04580904106572e-06
Best-20(val acc:0.25) | lr:0.0014388315064446943, weight decay:2.1869555317215038e-07

상위 5개까지의 결과를 보면,
Best-1(val acc:0.77) | lr:0.00642956548737644, weight decay:3.9335005750240353e-05
Best-2(val acc:0.76) | lr:0.009800708251553235, weight decay:4.32104341501499e-05
Best-3(val acc:0.74) | lr:0.008080563079160151, weight decay:4.032225845552401e-07
Best-4(val acc:0.74) | lr:0.008658154225122113, weight decay:1.6387860601920888e-08
Best-5(val acc:0.73) | lr:0.007174090437865117, weight decay:3.3679931489953985e-05
학습률은 0.001 ~ 0.1,  가중치 감소 계수는 10^-8 ~ 10^-5에 분포하고 있다.
이렇게 줄어든 범위로 똑같은 작업을 반복한다.
"""
