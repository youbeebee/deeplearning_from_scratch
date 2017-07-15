import numpy as np
'''
손실 함수loss function : 신경망 성능의 '나쁨'을 나타내는 지표.
일반적으로 평균 제곱 오차와 교차 엔트로피 오차를 사용
'''


# 4.2.1 평균 제곱 오차
# E = 1/2 * ∑ _k (yk-tk)²
# yk : 신경망의 출력
# tk : 정답 레이블
# k : 데이터의 차원 수
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


# 정답은 '2'
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# ex1 '2'일 확률이 가장 높다고 추정함(0.6)
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
mse = mean_squared_error(np.array(y), np.array(t))
print(mse)  # 0.0975
# ex2 '7'일 확률이 가장 높다고 추정함(0.6)
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
mse = mean_squared_error(np.array(y), np.array(t))
print(mse)  # 0.5975


# 4.2.2 교차 엔트로피 오차
# E = -∑ _k (tk * log(yk))
# log : 자연로그
# yk : 신경망의 출력
# tk : 정답 레이블(one-hot encoding)
# k : 데이터의 차원 수
# 실질적으로 정답일때의 추정의 자연로그를 계산하는 식이 됨
def cross_entropy_error(y, t):
    delta = 1e-7  # 0일때 -무한대가 되지 않기 위해 작은 값을 더함
    return -np.sum(t * np.log(y + delta))


# 동일한 계산
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
cee = cross_entropy_error(np.array(y), np.array(t))
print(cee)  # 0.510825457099
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
cee = cross_entropy_error(np.array(y), np.array(t))
print(cee)  # 2.30258409299
