import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from PIL import Image
from sklearn.linear_model import SGDClassifier
from keras.utils import np_utils
from tensorflow import keras
from tensorflow.keras import layers

def load_data(file_name, exist_label=False):
    images = []
    labels = []
    f = open(file_name, 'r')
    f_content = f.read().strip()
    f_lines = f_content.split('\n')
    [img_name, extension] = file_name.split('.')
    for line in f_lines:
        split = line.split(',')
        file_path = os.path.join(img_name,  split[0])
        img = np.asarray(Image.open(file_path))
        images.append(img)
        if exist_label is True:
            label = np.asarray(int(split[1]))
            labels.append(label)
    if exist_label is False:
        return images
    return images, labels


def devide(input):
    input = input.tolist()
    output = []
    for image in input:
        output.append([x / 255.0 for x in image])

    return np.asarray(output)


train_img, train_label = load_data('train.txt', exist_label=True)
valid_img, valid_label = load_data('validation.txt', exist_label=True)
test_img = load_data('test.txt')

train_img = np.asarray(train_img)
train_label = np.asarray(train_label)
valid_img = np.asarray(valid_img)
valid_label = np.asarray(valid_label)
test_img = np.asarray(test_img)

N, Nx_shape, Ny_shape = train_img.shape
train_img = np.reshape(train_img, (N, Nx_shape * Ny_shape))
N, Nx_shape, Ny_shape = valid_img.shape
valid_img = np.reshape(valid_img, (N, Nx_shape * Ny_shape))
N, Nx_shape, Ny_shape = test_img.shape
test_img = np.reshape(test_img, (N, Nx_shape * Ny_shape))

train_images = devide(train_img)
validation_images = devide(valid_img)
test_images = devide(test_img)

X, dim_x = train_images.shape
train_images = train_images.reshape((X, 50, 50, 1))
X, dim_x = validation_images.shape
validation_images = validation_images.reshape((X, 50, 50, 1))
X, dim_x = test_images.shape
test_images = test_images.reshape((X, 50, 50, 1))


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(50, 50, 1)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
# model.add(Dense(10, activation="softmax"))
#
# model = keras.Sequential()
# model.add(keras.Input(shape=(250, 250, 3)))  # 250x250 RGB images
# model.add(layers.Conv2D(32, 5, strides=2, activation="relu"))
# model.add(layers.Conv2D(32, 3, activation="relu"))
# model.add(layers.MaxPooling2D(3))

model.summary()

# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='sigmoid'))
# model.add(layers.Dense(3))
#
#
model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
#
# model.fit(train_images, train_label, epochs=5, validation_data=(validation_images, valid_label))
model.fit(
    train_images,
    train_label,
    batch_size=64,
    epochs=2,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(validation_images, valid_label),
)
#
# # train_label = np_utils.to_categorical(train_label, 10)
# # valid_label = np_utils.to_categorical(valid_label, 10)
# #
# # model = models.Sequential()
# # model.add(layers.Convolution2D(64, 5, 5, activation='relu', input_shape=(50, 50, 1)))
# # model.add(layers.Convolution2D(64, 5, 5, activation='relu'))
# # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# # model.add(layers.Dropout(0.25))
# #
# # model.add(layers.Flatten())
# # model.add(layers.Dense(128, activation='relu'))
# # model.add(layers.Dropout(0.25))
# # model.add(layers.Dense(10, activation='softmax'))
# #
# # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# #
# # model.fit(train_images, train_label, epochs=50,  batch_size=16)
#
test_loss, test_acc = model.evaluate(validation_images, valid_label, verbose=2)

print("Validation data accuracy:", test_acc)
print("Validation data loss", test_loss)
#
# probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# predictions = probability_model.predict(test_images)
#
#
# f = open("test.txt", "r")
# lines = f.readlines()
# lines = [line.strip().strip('"') for line in lines]
# ids = {"id": lines,
#        "label": [np.argmax(prediction) for prediction in predictions]
#        }
# df = pd.DataFrame(ids, columns=['id', 'label'])
# df.to_csv('submission.csv', index=False)
