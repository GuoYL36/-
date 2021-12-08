'''
利用tensorflow.keras来保存h5模型
'''

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
import tempfile





# load datasets
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
training_images = train_images/255.
test_images = test_images/255.

train_images = train_images.reshape(train_images.shape[0],28,28,1)
test_images = test_images.reshape(test_images.shape[0],28,28,1)


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = keras.Sequential([
    keras.layers.Conv2D(input_shape=(28,28,1),filters=8,
                        kernel_size=3,strides=2,activation="relu",name="Conv1"),
    keras.layers.Flatten(),
    keras.layers.Dense(10,activation=tf.nn.softmax,name="Softmax")
])

model.summary()

testing = False
epochs = 1
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=["accuracy"])
model.fit(train_images,train_labels,epochs=epochs)
test_loss,test_acc = model.evaluate(test_images,test_labels)

# MODEL_DIR = tempfile.gettempdir()
MODEL_DIR = "/home/xindun/guoyilin/models"
# MODEL_DIR = "D:\gyl\\tensorflow_keras\models"
version = 1
export_path = MODEL_DIR+"\\"+str(version)+"\\"+"saved_model.h5"
print(export_path)
tf.keras.models.save_model(
    model,export_path,overwrite=True,include_optimizer=True
)







