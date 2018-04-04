import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier


trw = 0.16
trh = 0.18
blw = 0.20
blh = 0.07

rowratio = 0.8567
colratio = 0.6716

#addr = input().strip()

form1 = cv2.imread('form5.jpg')
print(form1.shape)

form1 = cv2.resize(form1, (1654,2339), interpolation=cv2.INTER_LANCZOS4)
form2 = form1
form1 = cv2.cvtColor(form1, cv2.COLOR_BGR2GRAY)
# form = np.array(form)
# form = form.astype(np.uint8)
blur = cv2.GaussianBlur(form1, (3,3),0)
ret, form = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(form,cmap=plt.cm.gray,interpolation='nearest')
plt.show()

## corner detection
height, width = form.shape
print(height)
trindex = [-1, -1]
blindex = [-1, -1]
tlindex = [-1, -1]
tri = []
bli = []
trc = 0
blc = 0
tlc = 0
for i in range(600):
    trcount = 0
    blcount = 0
    tlcount = 0
    ltri = len(tri)
    if ltri<8:
        for j in range(int(width*trw)):
            if form[int(height*trh)+i,width-1-j] == 0:
                trcount += 1
                if trcount == 30:
                    tri.append([int(height*trh)+i,width-1-j+30])
                    break
            else:
                trcount = 0
    else:
        trc = 1
    lbli = len(bli)
    if lbli<8:
        for j in range(int(width*blw)):
            if form[height-int(height*blh)-i,j] == 0:
                blcount += 1
                if blcount == 30:
                    bli.append([height-int(height*blh)-i,j-30])
                    break
            else:
                blcount = 0
    else:
        blc = 1
    if tlc == 0:
        for j in range(int(width*blw)):
            if form[int(height*trh)+i,j] == 0:
                tlcount += 1
                if tlcount == 50:
                    tlindex = [int(height*trh)+i,j-50]
                    tlc = 1
                    break
            else:
                tlcount = 0
    if blc == 1 and trc == 1 and tlc == 1:
        break
print(bli)
print(tri)
for itm in range(len(bli)-1):
    if bli[itm][0] - 1 == bli[itm+1][0] and bli[itm][1]>=bli[itm+1][1]:
        continue
    else:
        blindex = bli[itm]
        break

for itm in range(1, len(tri)-1):
    if tri[itm][0] + 1 == tri[itm+1][0] and tri[itm][1]<=tri[itm+1][1]:
        continue
    else:
        trindex = tri[itm]
        break
blindex[0]-=1
trindex[0]+=1

brindex = [trindex[0] + blindex[0] - tlindex[0], blindex[1] + trindex[1] - tlindex[1]]
print(blindex, trindex, tlindex, brindex)

brien_drawing_lines = form2.copy()

cv2.line(brien_drawing_lines, tuple(trindex[::-1]), tuple(tlindex[::-1]), (0,255,0), 2)
cv2.line(brien_drawing_lines, tuple(tlindex[::-1]), tuple(blindex[::-1]), (255,0,0), 2)
cv2.line(brien_drawing_lines, tuple(blindex[::-1]), tuple(brindex[::-1]), (255,0,0), 2)
cv2.line(brien_drawing_lines, tuple(brindex[::-1]), tuple(trindex[::-1]), (255,0,0), 2)

plt.imshow(brien_drawing_lines, cmap=plt.cm.gray)
plt.show()

pts1 = np.float32([tlindex[::-1], trindex[::-1], brindex[::-1], blindex[::-1]])
pts2 = np.float32([[0, 0], [rowratio*width, 0], [rowratio*width, colratio*height], [0, colratio*height]])

M, s = cv2.findHomography(pts1, pts2)

dst = cv2.warpPerspective(form2, M, (int(rowratio*width), int(colratio*height)))

form2 = dst

plt.imshow(form2,cmap=plt.cm.gray,interpolation='nearest')
plt.show()

form2 = cv2.cvtColor(form2, cv2.COLOR_BGR2GRAY)
blur1 = cv2.GaussianBlur(form2, (5,5),0)
ret1, form = cv2.threshold(blur1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# plt.imshow(form,cmap=plt.cm.gray,interpolation='nearest')
# plt.show()


height, width = form.shape
## calculation of box size
bsize = (width-19*5-8)/20
bsize = int(bsize)+1
print(bsize)

## calculation of first box index
fboxindex = [4, 3]

itr_list = [[0, 20, 1, 'str'],[0, 20, 1, 'str'],[0, 20, 1, 'str'],[0, 8, 0, 'no.'],[15, 16, 1, 'str'],[0, 20, 1, 'str'],[0, 12, 1, 'no.'],[0, 20, 1, 'str'],
            [0, 20, 1, 'str'],[0, 20, 1, 'str'],[0, 20, 1, 'str'],[0, 6, 1, 'no.'],[0, 10, 1, 'no.']]
field_list = ['Name', 'Name of father', 'Name of Mother', 'Date of Birth', 'Gender', 'Occupation', 'Aadhaar Number',
              'Address1', 'Address2', 'City', 'State', 'Pincode', 'Mobile']
data ={}

pickle_in1 = open('newrandomforest40.pickle', 'rb')
clf1 = pickle.load(pickle_in1)
pickle_in2 = open('newrandomforestdigit40.pickle', 'rb')
clf2 = pickle.load(pickle_in2)

row = 0
for itr in range(len(itr_list)):
    field_data = ''
    for i in range(itr_list[itr][0], itr_list[itr][1]):
        space = 1
        temp = form[fboxindex[0] + int(row*1.895*bsize)+ row*11 + 3:fboxindex[0] + int(row*1.895*bsize) + bsize + row*11 -5,
               fboxindex[1] + i*bsize + int(i*5) + 2:fboxindex[1] + (i+1)*bsize + int(i*5) - 7]
        # temp = cv2.resize(temp, (28,28), interpolation=cv2.INTER_NEAREST)
        # blur = cv2.GaussianBlur(temp, (1, 1), 0)
        # ret1, temp = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # plt.imshow(temp, cmap=plt.cm.gray,interpolation='nearest')
        # plt.show()
        # print(temp)
        bimg = temp
        height, width = bimg.shape
        left = -1
        right = -1
        up = -1
        down = -1
        for h in range(height):
            for w in range(width):
                if bimg[h, w] == 0 and up == -1:
                    up = h
                if bimg[height - h - 1, w] == 0 and down == -1:
                    down = height - h - 1
        for w in range(width):
            for h in range(height):
                if bimg[h, w] == 0 and left == -1:
                    left = w
                if bimg[h, width - 1 - w] == 0 and right == -1:
                    right = width - w - 1
                if left > -1 and right > -1 and up > -1 and down > -1:
                    break
            if left > -1 and right > -1 and up > -1 and down > -1:
                break
        if left > -1 and right > -1 and up > -1 and down > -1:
            bimg = bimg[up:down + 1, left:right + 1]
            bimg = cv2.resize(bimg, (28, 28), interpolation=cv2.INTER_NEAREST)
            space = 0
            # plt.imshow(bimg, cmap=plt.cm.gray_r,interpolation='nearest')
            # plt.show()
        temp1 = np.array(temp)
        hght, wdth = temp1.shape
        temp1 = np.reshape(temp1, [1, hght*wdth])
        if np.sum(temp1) >= hght*(wdth-1)*255 :
            space = 1
        if space == 1:
            field_data += ' '
        else:
            pred = np.array(bimg)
            pred = np.reshape(pred, [1,784])
            label = []
            if itr_list[itr][3] == 'str':
                label = clf1.predict(pred)
            elif itr_list[itr][3] == 'no.':
                label = clf2.predict(pred)
            field_data += chr(int(label[0]))

    field_data = field_data.strip()
    if field_list[itr] == 'Address2':
        data[field_list[itr-1]] += ' ' + field_data
    else:
        data[field_list[itr]] = field_data
    row += itr_list[itr][2]

print(data)
