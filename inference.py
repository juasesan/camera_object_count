import pickle
import tensorflow as tf
import numpy as np
from utils import load_image

# Load image
img_rows, img_cols = 224, 224 # Resolution of inputs
image_path = './Images/bottle.jpg'
input_tensor = load_image(image_path,img_rows, img_cols)

# Load class labels
meta_path = './Data/cifar-100/meta'
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

class_labels = meta['fine_label_names']

# Load the model architecture from the JSON file
json_file = open('./Model/cifar100_resnet50.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(loaded_model_json)

# Load the model weights from the H5 file
model.load_weights('./Model/cifar100_resnet50.h5')


def main():
    # Make prediction
    predictions = model.predict(input_tensor)

    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = class_labels[predicted_class_index]

    print(np.max(predictions[0]))
    print(f'The predicted class is: {predicted_class_label}')


if __name__ == '__main__':
    main()


