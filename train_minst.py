import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()

x_train = (x_train)/255
x_test = (x_test)/255

y_tmp = np.zeros((y_train.size, y_train.max() + 1), dtype=int)
y_tmp[np.arange(y_train.size), y_train] = 1
y_train = y_tmp
y_tmp = np.zeros((y_test.size, y_test.max() + 1), dtype=int)
y_tmp[np.arange(y_test.size), y_test] = 1
y_test = y_tmp


x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)

x_train = 1 - x_train
x_test = 1 - x_test
# x_train = x_train.numpy()
# x_test = x_test.numpy()
# print(type(x_train))
# print(x_train.shape)
# for ll in ((x_train[0]).reshape(28,28)):
#     for lll in ll:
#         print("{:.4f}".format(lll), end=" ")
#     print()

# print("######################")

# x_train = np.load(os.path.join('data', 'X_split_train.npy'), mmap_mode='r')
# y_train = np.load(os.path.join('data', 'Y_split_train.npy'))
# x_test = np.load(os.path.join('data', 'X_split_test.npy'), mmap_mode='r')
# y_test = np.load(os.path.join('data', 'Y_split_test.npy'))

# x_train = np.asarray(x_train)
# x_test = np.asarray(x_test)


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

BATCH_SIZE = 64
train_dataset_batch = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset_batch = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)


model = models.Sequential()
model.add(layers.Conv2D(6, 5, activation='tanh', input_shape=x_train.shape[1:]))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(16, 5, activation='tanh'))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(120, 4, activation='tanh'))
model.add(layers.Flatten())
model.add(layers.Dense(84, activation='tanh'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=opt, 
              loss=losses.categorical_crossentropy, 
              metrics=['accuracy'])

history = model.fit(train_dataset_batch, 
                    batch_size=BATCH_SIZE, 
                    epochs=10, 
                    validation_data=validation_dataset_batch)

model.evaluate(x_test, y_test)
