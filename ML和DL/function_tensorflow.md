# tensorflow中的函数使用

- [content](#)
	- [sequence_mask](#sequence_mask函数)
	- [乘法](#直接乘和matmul和dot)

## sequence_mask函数
> tf.sequence_mask(lengths, maxlen=None, dtype=tf.bool, name=None)
> 返回值：布尔型mask张量，shape=(len(lengths), maxlen)

+ 示例
	+ 
	```python
	tf.sequence_mask([1, 3, 2], 5) = 
	[[True, False, False, False, False],      # 1表示第一个为True，其余为False
	 [True, True, True, False, False],        # 3表示前3个为True，其余为False
	 [True, True, False, False, False]]       # 2表示前2个为True，其余为False
		
	```
----	

## 直接乘和matmul和dot

+ 直接乘X
	
	```python
	v1 = tf.constant([[1,2],[4,5]])       
	v2 = tf.constant([[3,7],[6,1]])
	v1 x V2 =                             # 对应位置相乘
	[[3, 14],
	 [24, 5]]
	
	```

+ matmul
	
	```python
	v1 = tf.constant([[1,2],[4,5]])
	v2 = tf.constant([[3,7],[6,1]])	
	tf.matmul(v1, v2) =                    # 矩阵乘法，两个参数的维度必须对应（v1的列和v2的行相等）
	[[15, 9],
	 [42, 33]]
		
	```

+ tensordot
	
	```python
		
	v1 = tf.constant([[1,2],[4,5]])
	v2 = tf.constant([[3,7],[6,1]])
	
	# 示例1：当参数a和参数b是矩阵(2阶)时，axes=1相当于矩阵乘法。
	tf.tensordot(v1, v2, 1) =         # 矩阵乘法
	[[15, 9],
	 [42, 33]]
	 
	# 示例2：当参数a和参数b是矩阵(2阶)时，axes=[[1],[0]]相当于矩阵乘法。
	# 下面[[1],[0]]的结果和[1,0]的结果一样
	tf.tensordot(v1, v2, [[1],[0]]) =         # 矩阵乘法
	[[15, 9],
	 [42, 33]]		
	
	# 示例3：当参数axes是标量N时, 按顺序对v1的最后N个轴和v2的前N个轴进行相应位置相乘并求和.
	tf.tensordot(v1, v2, 0) =
	[
	 [
	  [[3, 7],
	   [6, 1]],           # v1的第一个元素和v2的所有元素对应位置相乘
	  [[6, 14],
	   [12, 2]]           # v1的第二个元素和v2的所有元素对应位置相乘
	 ],
	 [
	  [[12, 28],
	   [24, 4]],          # v1的第三个元素和v2的所有元素对应位置相乘
	  [[15, 35],
	   [30, 5]]           # v1的第四个元素和v2的所有元素对应位置相乘
	 ]
	]
	
	# 示例4：当参数axes是list或shape为(2,k)的tensor
	# 下面[0, 1]的结果和[[0],[1]]的结果一样
	tf.tensordot(v1, v2, [0, 1]) =        # v1的每一列(对应0)和v2的每一行(对应1)进行矩阵乘法（线代里矩阵乘法是v1的每一行和v2的每一列进行矩阵乘法）
	[[31, 10],       # [1*3+4*7, 1*6+4*1]
	 [41, 17]]       # [2*3+5*7, 2*6+5*1]

	```

+ tf.keras.backend.dot
	
	```python
	# 示例1：
	v1 = tf.constant([[1,2],[4,5]])
	v2 = tf.constant([[3,7],[6,1]])
	tf.keras.backend.dot(v1, v2) =         # 矩阵乘法
	[[15, 9],
	 [42, 33]]
	 
	# 示例2：
	v3=tf.constant([[1,2,3],[1,2,3]])        # shape: (2, 3)
	v4=tf.constant([[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]],
	                [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]],
					[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]],
					[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]])      # shape: (4, 3, 5)
	
	=
	[[[ 6, 12, 18, 24, 30],
      [ 6, 12, 18, 24, 30],
      [ 6, 12, 18, 24, 30],
      [ 6, 12, 18, 24, 30]],

     [[ 6, 12, 18, 24, 30],
      [ 6, 12, 18, 24, 30],
      [ 6, 12, 18, 24, 30],
      [ 6, 12, 18, 24, 30]]]           # shape: (2, 4, 5)
	
	```
	+ ![示例2](https://i.loli.net/2019/06/16/5d06151e7d80479439.jpg)

----

# tensorboard可视化
## 介绍
+ 简介：TensorBoard是tensorflow官方推出的可视化工具，它可以将模型训练过程中的各种汇总数据展示出来，包括标量(Scalars)、图片(Images)、音频(Audio)、计算图(Graphs)、数据分布(Distributions)、直方图(Histograms)和潜入向量(Embeddigngs)。
+ 作用：tensorflow代码执行过程是先构建图，然后在执行，所以对中间过程的调试不太方便；除此之外，在使用tensorflow训练大型深度学习神经网络时，中间的计算过程可能非常复杂，因此为了理解、调试和优化网络，可以使用TensorBoard观察训练过程中的各种可视化数据。
+ 可视化步骤
    + 建立graph；
    + 确定在graph中哪些节点放置summary操作记录信息
        + tf.name_scope：限制变量的作用域
        + tf.summary.scalar：用来记录标量，比如：学习率、目标函数等；
        + tf.summary.histogram：用来记录数据的直方图，比如：激活的分布、梯度权重的分布
        + tf.summary.distribution：用来记录数据的分布图
        + tf.summary.image：用来记录图像数据
    + 合并及存储summary信息
        + merged = tf.summary.merge_all()：利用这个可以将上述所有的summary节点合并为一个节点
        + writer = tf.summary.FileWriter('./logs',sess.graph)：保存记录数据
        + summary = sess.run([merged],feed_dict=train_feed_dict)
        + writer.add_summary(summary,e)：写入log文件，e是迭代数

    + web端可视化
        + tensorboard --logdir=文件路径
## 实例
```python
import tensorflow as tf

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)  # 参数1：变量名字；参数2：图片数据；参数3：图片展示的数量

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)    
    

def variable_summaries(var):  # 记录每次迭代后的数据
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      # 计算参数的均值，并使用tf.summary.scaler记录
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)

      # 计算参数的标准差
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      # 用直方图记录参数的分布
      tf.summary.histogram('histogram', var)  

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # 设置命名空间
    with tf.name_scope(layer_name):
      # 调用之前的方法初始化权重w，并且调用参数信息的记录方法，记录w的信息
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      # 调用之前的方法初始化权重b，并且调用参数信息的记录方法，记录b的信息
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      # 执行wx+b的线性计算，并且用直方图记录下来
      with tf.name_scope('linear_compute'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('linear', preactivate)
      # 将线性输出经过激励函数，并将输出也用直方图记录下来
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)

      # 返回激励层的最终输出
      return activations

hidden1 = nn_layer(x, 784, 500, 'layer1')
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

with tf.name_scope('loss'):
    # 计算交叉熵损失（每个样本都会有一个损失）
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      # 计算所有样本交叉熵损失的均值
      cross_entropy = tf.reduce_mean(diff)

tf.summary.scalar('loss', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(
        cross_entropy)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      # 分别将预测和真实的标签中取出最大值的索引，弱相同则返回1(true),不同则返回0(false)
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      # 求均值即为准确率
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)



## 合并summary operation
# summaries合并
merged = tf.summary.merge_all()
# 写到指定的磁盘路径中
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')
# 运行初始化所有变量
tf.global_variables_initializer().run()

def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
      xs, ys = mnist.train.next_batch(100)
      k = dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

for i in range(max_steps):
    if i % 10 == 0:  # 记录测试集的summary与accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # 记录训练集的summary
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
train_writer.close()
test_writer.close()
```
----








































