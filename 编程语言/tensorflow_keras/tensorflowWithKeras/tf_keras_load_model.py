'''
利用tensorflow.keras来加载h5模型
'''


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


# load datasets
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

training_images = train_images/255.
test_images = test_images/255.

train_images = train_images.reshape(train_images.shape[0],28,28,1)
test_images = test_images.reshape(test_images.shape[0],28,28,1)
test_images = np.array(test_images,dtype=np.float32)
print("test datasets number: ",test_images.shape)

MODEL_DIR = "/models/1/saved_model.h5"

model = tf.keras.models.load_model(MODEL_DIR)
y_pred = model.predict(test_images[:500])
print(y_pred[0], test_labels[0])



