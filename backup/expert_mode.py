
import math
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, InputLayer, Dropout, Conv1D, Flatten, Reshape, MaxPooling1D, BatchNormalization,
    Conv2D, GlobalMaxPooling2D, Lambda, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras import layers, models, losses

sys.path.append('./resources/libraries')
import ei_tensorflow.training

WEIGHTS_PATH = './transfer-learning-weights/edgeimpulse/MobileNetV1.0_1.96x96.grayscale.bsize_96.lr_0_05.epoch_363.val_accuracy_0.14.hdf5'

# Download the model weights
root_url = 'https://cdn.edgeimpulse.com/'
p = Path(WEIGHTS_PATH)
if not p.exists():
    print(f"Pretrained weights {WEIGHTS_PATH} unavailable; downloading...")
    if not p.parent.exists():
        p.parent.mkdir(parents=True)
    weights_data = requests.get(root_url + WEIGHTS_PATH[2:]).content
    with open(WEIGHTS_PATH, 'wb') as f:
        f.write(weights_data)
    print(f"Pretrained weights {WEIGHTS_PATH} unavailable; downloading OK")
    print("")

INPUT_SHAPE = (32, 32, 1)


# base_model = tf.keras.applications.MobileNet(
#     input_shape = INPUT_SHAPE,
#     weights = WEIGHTS_PATH,
#     alpha = 0.1
# )

# base_model.trainable = False

# model = Sequential()
# model.add(InputLayer(input_shape=INPUT_SHAPE, name='x_input'))
# # Don't include the base model's top layers
# last_layer_index = -5
# model.add(Model(inputs=base_model.inputs, outputs=base_model.layers[last_layer_index].output))
# model.add(Reshape((-1, model.layers[-1].output.shape[3])))
# model.add(Dropout(0.1))
# model.add(Flatten())
# model.add(Dense(classes, activation='softmax'))
model = models.Sequential()
model.add(layers.Conv2D(6, 5, activation='tanh', input_shape=X_train.shape[1:]))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(16, 5, activation='tanh'))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(120, 4, activation='tanh'))
model.add(layers.Flatten())
model.add(layers.Dense(84, activation='tanh'))
model.add(layers.Dense(10, activation='softmax'))



BATCH_SIZE = 32
# EPOCHS = args.epochs or 2
EPOCHS = 10
LEARNING_RATE = args.learning_rate or 0.0001
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)
callbacks.append(BatchLoggerCallback(BATCH_SIZE, train_sample_count, epochs=EPOCHS))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, verbose=2, callbacks=callbacks)

print('')
print('Initial training done.', flush=True)

# How many epochs we will fine tune the model
FINE_TUNE_EPOCHS = 10
# What percentage of the base model's layers we will fine tune
FINE_TUNE_PERCENTAGE = 65

print('Fine-tuning best model for {} epochs...'.format(FINE_TUNE_EPOCHS), flush=True)

# Load best model from initial training
model = ei_tensorflow.training.load_best_model(BEST_MODEL_PATH)

# Determine which layer to begin fine tuning at
model_layer_count = len(model.layers)
fine_tune_from = math.ceil(model_layer_count * ((100 - FINE_TUNE_PERCENTAGE) / 100))

# Allow the entire base model to be trained
model.trainable = True
# Freeze all the layers before the 'fine_tune_from' layer
for layer in model.layers[:fine_tune_from]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000045),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(train_dataset,
                epochs=FINE_TUNE_EPOCHS,
                verbose=2,
                validation_data=validation_dataset,
                callbacks=callbacks,
                class_weight=None
            )
 