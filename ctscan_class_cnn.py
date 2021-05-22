import numpy as np
import os
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet121
import seaborn as sn
classes = ['n', 'a', 'v']

'''
Clasa modelului meu, am implementat astfel pentru a modifica parametrii cu usurinta in apelare
In apelarea modelului functia initializeaza citirea din fisier, implicit, incarcarea datelor de antrenare, testare, validare
'''


class MyModel:
    def __init__(self, img_size=(50, 50), epochs=10, batch=36, learning_r=0.01, rgb_channel=3, optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy', class_dim=len(classes)):
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
            f_lines = f.read().strip().split('\n')
            for line in f_lines:
                composed = line.split(',')
                file_path = os.path.join(file.replace('.txt', ''), composed[0])
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

    '''
    Functia convert_input_shape(self, rgb_channel) verifica channelul pozei. In cazul cand channelul este diferit de 1, poza are 3 channele, deci trebuie updatat formatul input_shapeului care urmeaza
    sa-l punem in modelul clasificator.
    '''

    def convert_input_shape(self, rgb_channel):
        if int(rgb_channel) != 1:
            formatted_shape = list(self.input_shape)
            formatted_shape.append(rgb_channel)
            return tuple(formatted_shape)
        formatted_shape = list(self.input_shape)
        formatted_shape.append(rgb_channel)
        return tuple(formatted_shape)

    '''
    Functia de normalizare a datelor, verifica daca imaginea este alb-negru si daca da, adauga in shapeul listelor de imagini, channelul 1, corespunzator gri.
    Functia mai imparte datele la 255 pentru a normaliza datele, adica a incadra pixelii intre [0, 1].  
    Functia augumenteaza datele de train si validare cu rotatii de 45 de grade si flip orizontal intr-un batch de 32.
    '''
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

        '''DOAR PENTRU CATEGORICAL CROSSENTROPY'''

        # self.train_label = tf.keras.utils.to_categorical(self.train_label, self.class_dim)
        # self.valid_label = tf.keras.utils.to_categorical(self.valid_label, self.class_dim)

        self.datagen = ImageDataGenerator(
            rotation_range=45,
            horizontal_flip=True,
        )

        self.datagen.fit(self.train_img)

    '''
    Functie cu scop de initializare a unei "baze" de model pe care urmeaza sa ne construim modelul.
    Inghetam toate straturile de pe acest model, exceptand ultimele 3.
    '''

    def pre_train_wh_fine_tuning(self):
        base = DenseNet121(input_shape=self.input_shape, weights='imagenet', include_top=False)
        for layer in base.layers:
            layer.trainable = False
        for layer in base.layers[-3:]:
            layer.trainable = True
        return base

    '''
    Functia care construieste modelul.
    In prima incercare modelul se folosea de datele deja antrenate. In a doua incercare modelul este reprezentat de o retea convolutionara de 4 straturi de convolutie.
    '''

    def build_model_architecture(self):
        # base = self.pre_train_wh_fine_tuning()
        # self.model = models.Sequential()
        # self.model.add(base)
        # self.model.add(layers.Flatten())
        # self.model.add(layers.Dense(256, activation='relu'))
        # self.model.add(layers.Dense(64, activation='relu'))
        # self.model.add(layers.Dense(3, activation='softmax'))
        self.model = models.Sequential()
        self.model.add(layers.Convolution2D(16, 3, activation='relu', input_shape=self.input_shape))

        self.model.add(layers.Convolution2D(32, 3, activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(layers.Convolution2D(64, 3, activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(layers.Convolution2D(128, 3, activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(3, activation='softmax'))

    '''
    Functia afiseaza "summary" pentru modelul curent
    '''

    def show_summary(self):
        try:
            self.model.summary()
        except Exception as e:
            print(e)
            print("[SUMMARY] Model not yet built!")

    '''
    Functia care compileaza modelul curent
    '''

    def compile_model(self):
        try:
            self.model.compile(optimizer=self.opti, loss=self.loss, metrics=[self.metrics])
        except Exception as e:
            print(e)
            print("[COMPILE] Model not yet built!")

    '''
    Functia care antreneaza modelul curent
    Am lasat comentata acel rand in caz ca vrem sa antrenam modelul fara augumentare.
    '''

    def fit_model(self):
        try:

            # self.progress = self.model.fit(self.train_img, self.train_label, batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_data=(self.valid_img, self.valid_label))
            self.progress = self.model.fit(
                self.datagen.flow(self.train_img, self.train_label, batch_size=self.batch_size),
                batch_size=self.batch_size,
                validation_data=self.datagen.flow(self.valid_img, self.valid_label, batch_size=self.batch_size),
                epochs=self.epochs
                )

        except Exception as e:
            print(e)
            print("[FIT] Model not yet built!")

    '''
    Functia care se ocupa cu evaluarea modelului
    '''

    def evaluate_model(self):
        try:
            test_loss, test_acc = self.model.evaluate(self.valid_img, self.valid_label, verbose=2)
            print("Validation accuracy: ", test_acc)
            print("Validation loss: ", test_loss)
        except Exception as e:
            print(e)
            print("[EVALUATE] Model not yet built!")

    '''
    Functia care returneaza predictiile pentru a le pune in fisierul .csv.
    Aceasta functie se mai ocupa si de crearea matricei de confuzie
    '''

    def predict_model(self, on=None):
        try:
            if on is None:
                on = self.test_img
            probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
            predictions = probability_model.predict(on)
            return predictions
        except Exception as e:
            print(e)
            print("[PREDICT] Model not yet built!")

    '''
    Functia care creaza matricea de confuzie si o afiseaza
    '''

    def confusion(self):
        pred = self.predict_model(self.valid_img)
        d_predict = []
        for predict in pred:
            d_predict.append(np.argmax(predict))
        d_predict = np.asarray(d_predict)
        conf_matr = pd.crosstab(d_predict, self.valid_label, rownames=['actual'], colnames=['predicted'])
        sn.heatmap(conf_matr, annot=True)
        plt.show()

    '''
    Functia care incarca rezultatele intr-un fisier .csv
    '''

    def load_to_csv(self):
        f = open("test.txt", "r")
        imgs = f.readlines()
        img_name = []
        for img in imgs:
            img_name.append(img.strip().strip('"'))
        predictions = self.predict_model()
        label = []
        for prediction in predictions:
            label.append(np.argmax(prediction))
        data_obj = {"id": img_name, "label": label}
        df = pd.DataFrame(data_obj, columns=['id', 'label'])
        df.to_csv('submission.csv', index=False)

    '''
    Functia care ploteaza o imagine si o afiseaza
    '''

    def plot_img(self, index=1):
        clas = ["native", "arterial", "venous"]
        x = self.train_img
        y = self.train_label
        plt.figure(figsize=(15, 2))
        plt.imshow(x[index])
        plt.xlabel(clas[y[index]])
        plt.show()

    '''
    O functie ajutatoare ( pentru debug ) pentru printarea shapeului imaginii
    '''

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

    '''
    Functia care ruleaza modelul cu tot ce ne trebuie.
    '''

    def run_model(self):
        self.normalize()
        self.build_model_architecture()
        self.compile_model()
        self.show_summary()
        self.fit_model()
        self.confusion()
        self.evaluate_model()
        # self.plot_img(self.train_img, self.train_label, 1)  # example of img plotted
        # self.load_to_csv()


'''
Declararea modelului si rularea acestuia cu argumente custom.
'''
my_model = MyModel(epochs=20, batch=64)
my_model.run_model()
