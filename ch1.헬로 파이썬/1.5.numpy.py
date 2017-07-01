import numpy as np

# 1.5.1 넘파이 가져오기
# 1.5.2 넘파이 배열 생성하기
x = np.array([1.0, 2.0, 3.0])
print(x)  # [1. 2. 3.]
print(type(x))  # <class 'numpy.ndarray'>

# 1.5.3 넘파이의 산술 연산
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
print(x + y)  # 원소별 덧셈
print(x - y)
print(x * y)  # 원소별 곱셈(element-wise product)
print(x / y)
'''
[ 3.  6.  9.]
[-1. -2. -3.]
[  2.   8.  18.]
[ 0.5  0.5  0.5]
'''
print(x / 2.0)  # [ 0.5  1.   1.5]

# 1.5.4 넘파이의 N차원 배열
A = np.array([[1, 2], [3, 4]])
print(A)
'''
[[1 2]
 [3 4]]
'''
print(A.shape)  # 행렬의 형상 (2, 2)
print(A.dtype)  # 행렬에 담긴 원소의 자료형 int64

B = np.array([[3, 0], [0, 6]])
print(A + B)
print(A * B)
print(A * 10)
'''
[[ 4  2]
 [ 3 10]]
[[ 3  0]
 [ 0 24]]
[[10 20]
 [30 40]]
'''

# 1.5.5 브로드캐스트
A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
print(A * B)
'''
[[10 40]
 [30 80]]
'''

# 1.5.6 원소 접근
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
'''
[[51 55]
 [14 19]
 [ 0  4]]
 '''
print(X[0])  # [51 55]
print(X[0][1])  # 55
for row in X:
    print(row)

X = X.flatten()  # X를 1차원 배열로 변환(평탄화)
print(X)  # [51 55 14 19  0  4]
print(X[np.array([0, 2, 4])])  # 인덱스가 0,2,4인 원소 얻기 [51 14  0]
print(X > 15)  # [ True  True False  True False False]
print(X[X > 15])  # [51 55 19]
