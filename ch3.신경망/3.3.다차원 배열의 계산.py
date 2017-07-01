import numpy as np

# 3.3.1 다차원 배열
A = np.array([1, 2, 3, 4])
print(A)  # [1 2 3 4]
print(np.ndim(A))  # 1
print(A.shape)  # (4,)

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(np.ndim(B))  # 2
print(B.shape)  # (3, 2)

# 3.3.2 행렬의 내적(행렬 곱/스칼라 곱)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(np.dot(A, B))
'''
[[19 22]
 [43 50]]
'''

A = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
B = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
print(np.dot(A, B))  # (2, 2)
'''
[[22 28]
 [49 64]]
'''

X = np.array([1, 2])  # (2,)
W = np.array([[1, 3, 5], [2, 4, 6]])  # (2, 3)
Y = np.dot(X, W)
print(Y)  # [5 11 17]
