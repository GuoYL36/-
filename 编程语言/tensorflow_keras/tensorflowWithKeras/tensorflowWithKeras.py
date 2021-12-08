'''
tensorflow.keras
1、构建模型
2、权值共享
3、多输入多输出模型——tensorflow写法
4、多输入多输出模型——keras写法
'''


import tensorflow as tf
import numpy as np

##############################
# tensorflow2.0动态设置GPU占用
## 设置可见GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

conf = tf.compat.v1.ConfigProto()
conf.gpu_options.allow_growth = True  # 动态申请内存
session = tf.compat.v1.Session(config=conf)
tf.compat.v1.keras.backend.set_session(session)   # keras方法使得设置生效

##############################




################################################################################################
# # 全部流程主要以tensorflow为主，以tf.keras函数为辅
# img = tf.placeholder(tf.float32,shape=(None, 784))
# labels = tf.placeholder(tf.float32,shape=(None,10))
# x = tf.keras.layers.Dense(128,activation="relu")(img)
# x = tf.keras.layers.Dense(128,activation="relu")(x)
# x = tf.keras.layers.Dropout(0.6)(x)
# preds = tf.keras.layers.Dense(10,activation="softmax")(x)
#
# acc_value = tf.keras.metrics.categorical_accuracy(labels,preds)
# loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels,preds))
# optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
#
#
# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
# training_images = train_images/255.
# test_images = test_images/255.
#
# print(train_images.shape)
# print(test_labels.shape)
#
# train_images = train_images.reshape(train_images.shape[0],784)
# test_images = test_images.reshape(test_images.shape[0],784)
# def one_hot(data):
#     t = np.zeros((data.shape[0],10))
#     for i in range(data.shape[0]):
#         t[i,data[i]] = 1
#     return t
#
# train_labels = one_hot(train_labels)
# test_labels = one_hot(test_labels)
#
#
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(tf.keras.backend.learning_phase())
#     for i in range(10):
#         feed_dict = {img:train_images[:100],labels:train_labels[:100],
#                      tf.keras.backend.learning_phase():1}
#         _ = sess.run(optimizer, feed_dict=feed_dict)
#     print(tf.keras.backend.learning_phase())
#     feed_dict = {img:test_images[:100],labels:test_labels[:100],tf.keras.backend.learning_phase():0}
#     acc = sess.run(acc_value, feed_dict=feed_dict)
#     print(tf.keras.backend.learning_phase())
#     print("acc: ",acc)
################################################################################################

##############
# 权值共享
# lstm = tf.keras.layers.LSTM(32)
# x = tf.placeholder(tf.float32, shape=(None, 20, 64))
# y = tf.placeholder(tf.float32, shape=(None, 20, 64))
#
# x_encoded = lstm(x)
# y_encoded = lstm(y)
##############

############################################################################################
#############################################
# 多输入多输出模型
## tensorflow输入写法
# main_input = tf.placeholder(tf.float32, shape=(None,784),name="main_input")  # 输入1
# auxiliary_input = tf.placeholder(tf.float32, shape=(None,784), name="aux_input")  # 输入2
# labels = tf.placeholder(tf.float32,shape=(None,10))
#
# ## 网络1
# x = tf.keras.layers.Dense(128,activation="relu")(main_input)
# output1 = tf.keras.layers.Dense(10,activation="softmax")(x)    ## 输出1
#
# ## 网络2
# x = tf.keras.backend.concatenate([output1, auxiliary_input])
# x = tf.keras.layers.Dense(128,activation="relu")(x)
# x = tf.keras.layers.Dropout(0.5)(x)
# x = tf.keras.layers.Dense(128,activation="relu")(x)
# main_output = tf.keras.layers.Dense(10,activation="softmax",name="main_output")(x)
#
# ## 构建两个损失
# acc_value = tf.keras.metrics.categorical_accuracy(labels, main_output)
# loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, output1)) \
#        + tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, main_output))
# optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
# ##############################################
#
# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
# training_images = train_images/255.
# test_images = test_images/255.
#
# print(train_images.shape)
# print(test_labels.shape)
#
# train_images = train_images.reshape(train_images.shape[0],784)
# test_images = test_images.reshape(test_images.shape[0],784)
# def one_hot(data):
#     t = np.zeros((data.shape[0],10))
#     for i in range(data.shape[0]):
#         t[i,data[i]] = 1
#     return t
#
# train_labels = one_hot(train_labels)
# test_labels = one_hot(test_labels)
#
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(tf.keras.backend.learning_phase())
#     for i in range(10):
#         feed_dict = {main_input:train_images[:100],auxiliary_input:train_images[:100],labels:train_labels[:100],
#                      tf.keras.backend.learning_phase():1}
#         _ = sess.run(optimizer, feed_dict=feed_dict)
#     print(tf.keras.backend.learning_phase())
#     feed_dict = {main_input:test_images[:100],auxiliary_input:test_images[:100],labels:test_labels[:100],tf.keras.backend.learning_phase():0}
#     acc = sess.run(acc_value, feed_dict=feed_dict)
#     print(tf.keras.backend.learning_phase())
#     print("acc: ",acc)
#     print(test_labels[:100])


##############################################################################################





############################################################################################
#############################################
# 多输入多输出模型
######   keras写法
main_input = tf.keras.layers.Input(shape=(784,),dtype=tf.float32,name="main_input")
auxiliary_input = tf.keras.layers.Input(shape=(784,),dtype=tf.float32,name="auxiliary_input")

## 网络1
x = tf.keras.layers.Dense(128,activation="relu")(main_input)
dense_out = tf.keras.layers.Dense(10,activation="softmax")(x)
output1 = tf.keras.layers.Dense(10,activation="softmax")(dense_out)    ## 输出1

## 网络2
x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.concatenate(x))([dense_out, auxiliary_input])
x = tf.keras.layers.Dense(128,activation="relu")(x)
x = tf.keras.layers.Lambda(tf.keras.layers.Dropout(0.5))(x)
x = tf.keras.layers.Dense(128,activation="relu")(x)
main_output = tf.keras.layers.Dense(10,activation="softmax")(x)  # 输出2

model = tf.keras.models.Model(inputs=[main_input,auxiliary_input], outputs=[main_output,output1])
optimizer = tf.keras.optimizers.Adam(0.01)
model.compile(optimizer=optimizer,loss=tf.keras.losses.categorical_crossentropy,loss_weights=[1.0,1.0])


##
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

##
# model.fit([train_images[:100],train_images[:100]],[train_labels[:100],train_labels[:100]],
#           batch_size=32,epochs=10)
model.fit({"main_input":train_images[:100],"auxiliary_input":train_images[:100]},{"output1":train_labels[:100],"main_output":train_labels[:100]},batch_size=32,epochs=10)
y_pred = model.predict([test_images[:100],test_images[:100]],batch_size=100)
print(y_pred)

################################################################################################


