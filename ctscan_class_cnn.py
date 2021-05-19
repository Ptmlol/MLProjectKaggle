import numpy as np
import os
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet import DenseNet121


classes = ["native", "arterial", "venous"]


class MyModel:
    def __init__(self, img_size=(50, 50), epochs=15, batch=36, learning_r=0.01, rgb_channel=3, optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy', class_dim=len(classes)):
        self.img_size = img_size
        self.epochs = epochs
        self.batch_size = batch
        self.learning_rate = learning_r
        self.model = None
        self.input_shape = img_size
        self.input_shape = self.convert_input_shape(rgb_channel)
        self.gray = 1 if rgb_channel == 1 else False
        self.progress = None
        self.datagen = None
        self.opti = optimizer
        self.loss = loss
        self.metrics = metrics
        self.class_dim = class_dim
        self.width = img_size[0]
        self.height = img_size[1]

        def load(file, exist_label=False):
            images = []
            labels = []
            f = open(file, 'r')
            f_content = f.read().strip()
            f_lines = f_content.split('\n')
            dir_name = file.replace('.txt', '')
            for line in f_lines:
                composed = line.split(',')
                file_path = os.path.join(dir_name, composed[0])
                image = cv2.imread(file_path)
                images.append(cv2.resize(image, self.img_size))
                if exist_label:
                    labels.append(int(composed[1]))
            images = np.array(images)
            if not exist_label:
                return images
            labels = np.array(labels)
            return images, labels

        self.train_img, self.train_label = load('train.txt', True)  # returns np array
        self.valid_img, self.valid_label = load('validation.txt', True)  # returns np array
        self.test_img = load('test.txt')  # returns np array

    def convert_input_shape(self, rgb_channel):
        if int(rgb_channel) != 1:
            formatted_shape = list(self.input_shape)
            formatted_shape.append(rgb_channel)
            return tuple(formatted_shape)
        formatted_shape = list(self.input_shape)
        formatted_shape.append(rgb_channel)
        return tuple(formatted_shape)

    def normalize(self):
        if self.gray:
            self.train_img = self.train_img.reshape(self.train_img.shape[0], self.width, self.height, self.gray)
            self.test_img = self.test_img.reshape(self.test_img.shape[0], self.width, self.height, self.gray)
            self.valid_img = self.valid_img.reshape(self.valid_img.shape[0], self.width, self.height, self.gray)
        self.train_img = self.train_img.astype('float32')
        self.test_img = self.test_img.astype('float32')
        self.valid_img = self.valid_img.astype('float32')
        self.train_img = self.train_img / 255
        self.test_img = self.test_img / 255
        self.valid_img = self.valid_img / 255
        # self.train_label = tf.keras.utils.to_categorical(self.train_label, self.class_dim)
        # self.valid_label = tf.keras.utils.to_categorical(self.valid_label, self.class_dim)
        self.datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        self.datagen.fit(self.train_img)

    def pre_train_wh_fine_tuning(self):
        base = DenseNet121(input_shape=self.input_shape, weights='imagenet', include_top=False)
        for layer in base.layers[:3]:
            layer.trainable = False
        for layer in base.layers[-3:]:
            layer.trainable = True
        return base

    def build_model_architecture(self):
        base = self.pre_train_wh_fine_tuning()
        self.model = models.Sequential()
        # self.model.add(layers.Lambda(lambda x: backend.resize_images(x, height_factor=4, width_factor=4, data_format='channels_last')))
        self.model.add(base)
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dropout(0.25))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Dense(3, activation='softmax'))

    def show_summary(self):
        try:
            self.model.summary()
        except Exception as e:
            print(e)
            print("[SUMMARY] Model not yet built!")

    def compile_model(self):
        try:
            self.model.compile(optimizer=self.opti, loss=self.loss, metrics=[self.metrics])
        except Exception as e:
            print(e)
            print("[COMPILE] Model not yet built!")

    def fit_model(self):
        try:

            # self.model.fit(self.train_img, self.train_label, batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_data=(self.valid_img, self.valid_label))
            self.progress = self.model.fit(self.datagen.flow(self.train_img, self.train_label, batch_size=32), steps_per_epoch=len(self.train_img) // 32, epochs=self.epochs)
        except Exception as e:
            print(e)
            print("[FIT] Model not yet built!")

    def evaluate_model(self):
        try:
            test_loss, test_acc = self.model.evaluate(self.valid_img, self.valid_label, verbose=2)
            print("Validation accuracy: ", test_acc)
            print("Validation loss: ", test_loss)
        except Exception as e:
            print(e)
            print("[EVALUATE] Model not yet built!")

    def predict_model(self):
        try:
            probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
            predictions = probability_model.predict(self.test_img)
            return predictions
        except Exception as e:
            print(e)
            print("[PREDICT] Model not yet built!")

    def load_to_csv(self):
        f = open("test.txt", "r")
        lines = f.readlines()
        lines = [line.strip().strip('"') for line in lines]
        predictions = self.predict_model()
        ids = {"id": lines,
               "label": [np.argmax(prediction) for prediction in predictions]
               }
        df = pd.DataFrame(ids, columns=['id', 'label'])
        df.to_csv('submission.csv', index=False)

    @staticmethod
    def plot_img(x, y, index):
        plt.figure(figsize=(15, 2))
        plt.imshow(x[index])
        plt.xlabel(classes[y[index]])
        plt.show()

    def print_arrays(self):
        print("train_img")
        print(self.train_img.shape)
        print('\nvalid_img')
        print(self.valid_img.shape)
        print('\ntrain_label')
        print(self.train_label.shape)
        print('\nvalid_label')
        print(self.valid_label.shape)
        print("\ntest_img")
        print(self.test_img.shape)
        print("\ninput_shape")
        print(self.input_shape)

    def run_model(self):
        self.normalize()
        self.build_model_architecture()
        self.compile_model()
        self.show_summary()
        self.fit_model()
        self.evaluate_model()
        # self.plot_img(self.train_img, self.train_label, 1)  # example of img plotted
        self.load_to_csv()


my_model = MyModel(img_size=(32,32))
# my_model.normalize()
# my_model.data_augment()
# my_model.print_arrays()
my_model.run_model()

