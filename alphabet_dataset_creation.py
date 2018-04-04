import matplotlib.pyplot as plt
import numpy as np
import cv2, os
from datetime import datetime

start = datetime.now()

print(start)

data = np.array([])
label = np.array([])
count=0

for fol in range(7):
    for character in range(65, 91):
        addr = "E:/btp/by_field/hsf_"+str(fol)+"/upper/"+str(format(character, '02x'))
        print(addr)
        file_names = os.listdir(addr)
        count += len(file_names)
        lbl = [character]*len(file_names)
        label = np.append(label, lbl)
        for file_name in file_names:
            address = addr + "/" + file_name
            repgray = cv2.imread(address,0)
            blur = cv2.GaussianBlur(repgray, (5, 5), 0)
            r, repbin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
            data = np.append(data, repfinal)

data = np.reshape(data, [count, 784])
# temp = np.reshape(data[-500], [12, 16])
# plt.imshow(temp, cmap=plt.cm.gray_r, interpolation='nearest')
# plt.show()
np.save('alphabet_dataset', data)
np.save('alphabet_label', label)

stop = datetime.now()

print(stop-start)