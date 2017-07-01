import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# 1.6.1 단순한 그래프 그리기
# 데이터 준비
x = np.arange(0, 6, 0.1)
y = np.sin(x)

# 그래프 그리기
# plt.plot(x, y)
# plt.show()

# 1.6.2 pyplot의 기능
# 데이터 준비
x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

# 그래프 그리기
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos")
plt.xlabel("x")  # 축이름
plt.ylabel("y")
plt.title('sin & cos')
plt.legend()
plt.show()

# 1.6.3 이미지 표시하기

img = imread('../dataset/lena.png')

plt.imshow(img)
plt.show()
