'''
对不同层设置不同学习率
'''
import tensorflow as tf
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

print(tf.__version__)



x = tf.placeholder(dtype=tf.float32, shape=[None, 50], name="input")
y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="output")

w = tf.Variable(initial_value=tf.initializers.random_normal(mean=0.0,stddev=1.0)(shape=[50,10],dtype=tf.float32),trainable=True,name="w")
b = tf.Variable(initial_value=tf.zeros([10]),name="b")
x1 = tf.matmul(x,w)+b


w1 = tf.Variable(initial_value=tf.initializers.random_normal(mean=0.0,stddev=1.0)(shape=[10,2],dtype=tf.float32),trainable=True,name="w1")
b1 = tf.Variable(initial_value=tf.zeros([2]),name="b1")
out = tf.matmul(x1,w1)+b1


loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y, out))

var = tf.trainable_variables()
print(var)
var_list1 = var[:2]   # w和b
var_list2 = var[2:]   # w1和b1


opt1 = tf.train.AdamOptimizer(0.01)
opt2 = tf.train.AdamOptimizer(0.0001)
grads = tf.gradients(loss, var_list1+var_list2)
print(grads)
grads1 = grads[:2]
grads2 = grads[2:]

train_op1 = opt1.apply_gradients(zip(grads1,var_list1))
train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
train_op = tf.group(train_op1, train_op2)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    data = np.random.random((1000, 50))
    label = np.zeros((1000))

    label[:500] = 1

    df = np.concatenate((data, np.reshape(label,(-1,1))),axis=1)
    np.random.shuffle(df)

    def to_onehot(a):
        tmp = []
        for i in range(a.shape[0]):
            tmp1 = [0,0]
            tmp1[int(a[i])] = 1
            tmp.append(tmp1)
        return np.array(tmp)

    train_data = df[:900,:-1]
    train_label = df[:900, -1]
    train_label = to_onehot(train_label)

    val_data = df[900:,:-1]
    val_label = df[900:,-1]
    val_label = to_onehot(val_label)

    batch_size = 16
    epochs = 1
    for i in range(epochs*1000//batch_size):

        _ = sess.run(train_op, feed_dict={x:train_data[i*batch_size:(i+1)*batch_size,:], y:train_label[i*batch_size:(i+1)*batch_size,:]})

        if i % 50 == 0:
            loss1, g, bb, bb1 = sess.run([loss, grads, b, b1], feed_dict={x:val_data, y:val_label})
            print("step: %d | loss: %f"%(i, loss1))
            print(g[1],bb)
            print(g[3],bb1)
















