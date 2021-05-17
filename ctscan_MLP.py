import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier

# WEAK < 0.34 acc online, 0.49 brut


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


def normalize_data(train, test=None):
    scaler = preprocessing.StandardScaler()
    scaler.fit(train)
    scaler_training_data = scaler.transform(train)

    if test is not None:
        scaler_test_data = scaler.transform(test)
        return scaler_training_data, scaler_test_data

    return scaler_training_data


def normalizare_train_model(model, training_samples, labels, testing_data, testing_labels):
    training_samples, testing_data = normalize_data(training_samples, testing_data)
    model.fit(training_samples, labels)
    print('plain - acuratetea pe multimea de antrenare este ', model.score(training_samples, labels))
    print('acuratetea pe multimea de testare este ', model.score(testing_data, testing_labels))
    print('numarul de iteratii parcurse pana la convergenta %d' % model.n_iter_)


train_img, train_label = load_data('train.txt', exist_label=True)
valid_img, valid_label = load_data('validation.txt', exist_label=True)
test_img = load_data('test.txt')

train_img = np.asarray(train_img)
train_label = np.asarray(train_label)
valid_img = np.asarray(valid_img)
valid_label = np.asarray(valid_label)
test_img = np.asarray(test_img)

mlp_classifier_model = MLPClassifier(
    hidden_layer_sizes=(100, ),
    activation='relu',
    solver='sgd',
    alpha=0.0001,
    batch_size='auto',
    learning_rate='adaptive',
    learning_rate_init=0.0008,
    power_t=0.5,
    max_iter=300,
    shuffle=True,
    random_state=None,
    tol=0.0001,
    momentum=0.9,
    early_stopping=False,
    validation_fraction=0.1,
    n_iter_no_change=10
)

N, Nx_shape, Ny_shape = train_img.shape
train_img = np.reshape(train_img, (N, Nx_shape * Ny_shape))

N, Nx_shape, Ny_shape = valid_img.shape
valid_img = np.reshape(valid_img, (N, Nx_shape * Ny_shape))

N, Nx_shape, Ny_shape = test_img.shape
test_img = np.reshape(test_img, (N, Nx_shape * Ny_shape))


normalizare_train_model(mlp_classifier_model, train_img, train_label, valid_img, valid_label)


sg = SGDClassifier(random_state=42)
sg.fit(train_img, train_label)
pred = sg.predict(test_img)

f = open("test.txt", "r")
lines = f.readlines()
lines = [line.strip().strip('"') for line in lines]

ids = {"id": lines,
       "label": pred.tolist()
       }
df = pd.DataFrame(ids, columns=['id', 'label'])

df.to_csv('submission.csv', index=False)
