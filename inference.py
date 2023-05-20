import pickle
import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.layers import Input

from utils import load_image


# Load image
#img_rows, img_cols = 224, 224 # Resolution of inputs
image_path = './Images/orange.jpg'
# Load and preprocess the image
img = image.load_img(image_path, target_size=(224, 224))
x = image.img_to_array(img)
x = preprocess_input(x)
input_tensor = np.expand_dims(x, axis=0)

#input_tensor = load_image(image_path,img_rows, img_cols)

# Load class labels
'''meta_path = './Data/cifar-100/meta'
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

class_labels = meta['fine_label_names']'''
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())


# Load the model architecture from the JSON file
'''json_file = open('./Model/cifar100_resnet50.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(loaded_model_json)'''
model = ResNet50(weights='imagenet')

# Load the model weights from the H5 file
#model.load_weights('./Model/cifar100_resnet50.h5')


def main():
    # Make prediction
    predictions = model.predict(input_tensor)

    # Returns a list of tuples (class, description, probability)
    predicted_labels = decode_predictions(predictions, top=1)[0]

    # Extract the class label from each tuple
    class_labels = [label[1] for label in predicted_labels]

    print(class_labels)
    '''predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = imagenet_labels[predicted_class_index]

    print(np.max(predictions[0]))
    print(f'The predicted class is: {predicted_class_label}')'''


if __name__ == '__main__':
    main()


