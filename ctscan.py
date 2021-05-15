import os
import numpy as np
from PIL import Image


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
        images.append(np.array(Image.open(file_path)))
        if exist_label is True:
            labels.append(int(split[1]))
    if exist_label is False:
        return images
    return images, labels


train_img, train_label = load_data('train.txt', exist_label=True)
valid_img, valid_label = load_data('validation.txt', exist_label=True)
test_img = load_data('test.txt')

