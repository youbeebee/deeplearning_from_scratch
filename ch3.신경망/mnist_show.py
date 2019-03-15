import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from dataset.mnist import load_mnist

# 3.6.1 MNIST 이미지 확인해보기

sys.path.append(os.pardir)


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    plt.imshow(pil_img)
    plt.show()


(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5
print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 원래 이미지 모양으로 변형
print(img.shape)  # (28, 28)

img_show(img)
