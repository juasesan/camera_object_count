import cv2
import numpy as np

from keras.datasets import cifar100
from keras import backend as K
from keras.utils import np_utils


num_classes = 100

def load_cifar100_data(img_rows, img_cols):

    # Load cifar10 training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = cifar100.load_data()
    nb_train_samples = len(X_train)     # 3000 training samples
    nb_valid_samples = len(X_valid)     # 100 validation samples

    # Resize trainging images
    if K.image_dim_ordering() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_valid[:nb_valid_samples,:,:,:]])
    else:
        X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:nb_valid_samples,:,:,:]])

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], num_classes)
    Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)

    return X_train, Y_train, X_valid, Y_valid


def load_image(image_path, img_rows, img_cols):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_rows,img_cols))
    img = img / 255.0   # Normalization

    img = np.array(img)
    img = np.expand_dims(img, axis=0)   # Image tensor of shape (1,224,224,3)

    return img
