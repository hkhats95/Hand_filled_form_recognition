import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from mnist import MNIST
import os, struct
import cv2
from datetime import datetime

start = datetime.now()

img_name = os.path.join('samples','train-images.idx3-ubyte')
lbl_name = os.path.join('samples','train-labels.idx1-ubyte')

with open(lbl_name, 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    lbl = np.fromfile(flbl, dtype=np.uint8)

label = np.array(lbl)
label = label + 48

with open(img_name, 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), 784)

for ind in range(len(img)):
    rep = img[ind].reshape(28,28)
    #repgray = cv2.cvtColor(rep, cv2.COLOR_BGR2GRAY)
    rep1 = cv2.resize(rep, (128, 128), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(rep1, (5, 5), 0)
    r, repbin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    height, width = repbin.shape
    left = -1
    right = -1
    up = -1
    down = -1
    for i in range(height):
        for j in range(width):
            if repbin[i, j] == 0 and up == -1:
                up = i
            if repbin[height - i - 1, j] == 0 and down == -1:
                down = height - i - 1
            if repbin[j, i] == 0 and left == -1:
                left = i
            if repbin[j, width - 1 - i] == 0 and right == -1:
                right = width - i - 1
            if left > -1 and right > -1 and up > -1 and down > -1:
                break
        if left > -1 and right > -1 and up > -1 and down > -1:
            break

    repfinal = repbin[up:down + 1, left:right + 1]
    repfinal = cv2.resize(repfinal, (28, 28), interpolation=cv2.INTER_NEAREST)
    img[ind] = repfinal.reshape(1,784)
# print(label[0], imge[0])
# plt.imshow(imge[0].reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
# plt.show()
np.save('digits_dataset', img)
np.save('digits_label', label)

print(datetime.now()-start)