'''
    利用tf.keras.models.Model获取中间层输入
'''

import tensorflow as tf
import numpy as np


# 构建模型
main_input = tf.keras.layers.Input(shape=(784,),dtype=tf.float32,name="main_input")
x = tf.keras.layers.Dense(128,activation="relu")(main_input)
x = tf.keras.layers.Dense(64,activation="relu")(x)
x = tf.keras.layers.Dense(32,activation="relu")(x)
x = tf.keras.layers.Dense(16,activation="relu")(x)
dense_out = tf.keras.layers.Dense(10,activation="softmax")(x)
output = tf.keras.layers.Dense(10,activation="softmax")(dense_out)

model = tf.keras.models.Model(inputs=main_input, outputs=output)
optimizer = tf.keras.optimizers.Adam(0.01)
model.compile(optimizer=optimizer,loss=tf.keras.losses.categorical_crossentropy)

# 打印模型结构
print(model.summary())


## 训练模型
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
training_images = train_images/255.
test_images = test_images/255.

print(train_images.shape)
print(test_labels.shape)

train_images = train_images.reshape(train_images.shape[0],784)
test_images = test_images.reshape(test_images.shape[0],784)
def one_hot(data):
    t = np.zeros((data.shape[0],10))
    for i in range(data.shape[0]):
        t[i,data[i]] = 1
    return t

train_labels = one_hot(train_labels)
test_labels = one_hot(test_labels)


model.fit(train_images[:100],train_labels[:100],batch_size=32,epochs=10)
y_pred = model.predict([test_images[:100],test_images[:100]],batch_size=100)
print(y_pred)

# 利用tf.keras.models.Model函数构建输入到中间层的过程
middle_model = tf.keras.models.Model(inputs=main_input, outputs=model.get_layer("dense_3").output)
middle_model_predict = middle_model.predict(train_images[:10])
print("="*30)
print(middle_model_predict)









