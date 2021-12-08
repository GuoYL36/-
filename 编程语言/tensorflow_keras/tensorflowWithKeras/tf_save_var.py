'''
tensorflow保存中间变量
tensorflow加载中间变量
'''

import tensorflow as tf
import os

# # 保存变量
# v1 = tf.Variable(initial_value=tf.zeros(1),name="v1",dtype=tf.float32)
# v2 = tf.Variable(initial_value=tf.zeros(1),name="v2",dtype=tf.float32)
#
# inc_v1 = v1.assign(v1+1)
# dec_v2 = v2.assign(v2-1)
#
# model_path = "../models/2/model"
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#
#     saver = tf.train.Saver({"v1_name":v1,"v2_name":v2})
#     saver.save(sess,model_path)

# 加载变量

model_path = "../models/2/"

with tf.Session() as sess:

    # 通过网络图来获取变量
    ## 恢复网络图

    saver = tf.train.import_meta_graph("../models/2/model.meta")
    saver.restore(sess, tf.train.latest_checkpoint(model_path))

    # 直接打印已经保存的变量
    print(sess.run("v1_name:0"))

    # 通过图来获取变量
    graph = tf.get_default_graph()
    v = graph.get_tensor_by_name("v1_name:0")
    print(sess.run(v))

























