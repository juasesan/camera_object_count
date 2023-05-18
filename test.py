import tensorflow as tf
from sklearn.metrics import log_loss

from utils import load_cifar100_data


# Data Loading
img_rows, img_cols = 224, 224 # Resolution of inputs
channel = 3
num_classes = 100 
batch_size = 16 
nb_epoch = 50

X_train, Y_train, X_valid, Y_valid = load_cifar100_data()
del X_train, Y_train

# Model loading
model_path = './Model/cifar100_resnet50.h5'
model = tf.keras.models.load_model(model_path)


def main():
    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    scores = log_loss(Y_valid, predictions_valid)
    print("Cross-entropy loss score",scores)

    ## evaluate modelon test data:
    score = model.evaluate(X_valid, Y_valid, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))


if __name__ == '__main__':
    main()