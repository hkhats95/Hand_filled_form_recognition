import numpy as np

alpha = {}
digit = {}
lbl1 = np.load('new_alphabet_label.npy')
lbl2 = np.load('new_digits_label.npy')

a = lbl1.tolist()
b = lbl2.tolist()
for i in range(65, 91):
    alpha[chr(i)] = a.count(i)
for j in range(48,58):
    digit[chr(j)] = b.count(j)
print(alpha)
print(digit)