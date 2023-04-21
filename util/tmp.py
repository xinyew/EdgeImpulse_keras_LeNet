import numpy as np
import os, sys

x_train = np.load(os.path.join('data', 'X_split_train.npy'), mmap_mode='r')
y_train = np.load(os.path.join('data', 'Y_split_train.npy'))
x_test = np.load(os.path.join('data', 'X_split_test.npy'), mmap_mode='r')
y_test = np.load(os.path.join('data', 'Y_split_test.npy'))

# x_train = np.asarray(x_train)
# x_test = np.asarray(x_test)

print(y_train[int(sys.argv[1])])
with open('pixels_data_viz/viz', 'w') as f0:
    for ll in (x_train[int(sys.argv[1])].reshape(28,28)):
        for lll in ll:
            f0.write("{:.4f}".format(lll))
            f0.write(" ")
        f0.write("\n")

with open('pixels_data_viz/viz', 'r') as f:
    with open('pixels_data_viz/vizz', 'w') as f1:
        s = f.read()
        for line in s.splitlines():
            for ll in line.split(' '):
                if ll:
                    if float(ll) > 0.9:
                        f1.write('1.0000')
                    else:
                        f1.write('0.0000')
                f1.write(" ")
            f1.write("\n")
