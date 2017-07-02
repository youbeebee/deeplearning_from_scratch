import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image

# 3.6.1 MNIST 이미지 확인해보기


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,
    normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5
print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 원래 이미지 모양으로 변형
print(img.shape)  # (28, 28)

img_show(img)
