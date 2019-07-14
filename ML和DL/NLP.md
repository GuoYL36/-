## NLP

- [content]
	- [TFIDF](#TF-IDF)
	- [word2vec](#word2vec)
	- [Glove](#Glove)
	- [Fasttext](#Fasttext)
	- [为什么有了word2vec，还要Glove、ELMO等？](#为什么有了word2vec，还要Glove、ELMO等？)
	- [BPE](#BPE)
	- [SentencePiece](#SentencePiece)
	- [Attention的实现](#Attention的实现)
	- [Beam Search](#BeamSearch)
	
	
### TF-IDF
+ **IDF为什么用对数？**
	+ 涉及到信息论的知识，指IDF相当于一个权重，当该词越容易分辨，则相应权重也会特别大。
----

### word2vec
> 输入层+映射层+输出层
> 两种模型：skip-gram模型、CBOW模型

+ **损失函数**
	+ -log(P)，P为softmax函数

+ **skip-gram模型**
	+ 又称跳字模型，在n个word窗口内，由中心词预测n-1个背景词，要循环n-1遍。

+ **CBOW模型**
	+ 又称连续词袋模型，在n个word窗口内，由n-1个背景词预测中心词。输入层到映射层是将n-1个word的词向量相加。

+ **模型训练方法**
	+ 负采样
		+ 背景词是出现在中心词时间窗口内的词。噪声词是和中心词不同时出现在该时间窗口的词，噪声词由噪声词分布采样得到。噪声词分布可以是字典中的所有词的词频概率组成（一般是单字概率的3/4次方）。
	+ 层序softmax
		+ 构建huffman树(数据结构)
			+ 第一步，统计每个word的词频，根据词频对word从大到小排序，构建字典；第二步，从字典中选出2个词频最小的word作为叶子结点并构建父结点，从词典中将这2个word删除；第三步，从剩余的字典中选择词频最小的word作为叶子结点，并与刚才构建的父结点一起构建新的父结点，然后，将这个word从字典中删除；第四步，重复第3步直至字典为空，此时完成huffman树的构建。
		+ 原来的softmax中是从字典中选一个单词，而层序softmax的思路是将一次分类分解为多次分类，因此采用树的结构，而为什么采用Huffman树呢？是因为这样构建的话（根据词频构造），出现频率越高的词所经过的路径越短，从而使得所有单词的平均路径长度最短。
		+ Huffman树的节点含义：非叶子节点可以认为是神经网络中隐藏层参数（初始化长度为m的零向量），叶子节点为字典中的词向量（随机初始化），每个词具有唯一的编码。

+ **训练过程**
	+ 因为层序softmax的映射层到输出层是Huffman树，在训练过程中，对于CBOW来说，先将n-1个背景词向量相加然后作为映射层的输入，然后根据Huffman树计算每个非叶节点向量和输入向量的乘积再用sigmoid计算概率，依次递归下去直至叶节点，因为每个词都对应有唯一的一个编码，所以训练过程中的损失函数就是根据预测词的编码（分类的思想）累乘每次分叉时的概率，使得该概率值最大。

+ **为什么有了层序softmax后还要负采样的方法？**
	+ 虽然层序softmax可以提高模型的训练效率，但是如果训练样本中的中心词是一个**生僻词**，那么模型得到的Huffman编码就是一个很长的序列。
----

### Glove
+ 可以很好地表示词与词之间的类比关系;
+ 能够知道每个词的全局统计信息。
----

### Fasttext
+ Fasttest在word2vec基础上加入了n-gram的信息。解决oov问题。
+ 对于输入，也是将窗口内的word的词向量和n-gram的word的词向量相加。
----

### 为什么有了word2vec，还要Glove、ELMO等？
+ word2vec
	+ 学习到词之间的类比信息，适合局部间存在很强关联性的文本，但是缺乏词与词之间的共现信息。词义相近的词对贡献次数多，词义差得比较远的词对共现次数比较少，但是其实它们的区分度并不明显。
+ Glove
	+ 对word-pair共现进行建模，拟合不同word-pair的共现之间的差异。
	+ 相比于word2vec，Glove更容易并行化，速度更快。
+ ELMO
	+ 前两者学到某个词的词向量是固定的，不能很好处理一词多义。ELMO是基于整个语料训练的，而不是窗口，因而能更好地表达一个词。
+ Bert
	+ Bert中利用mask的方法有点像负采样，只不过Bert是基于整个语料做的，Bert主要是推翻了现有的对于某个任务必须特定的模型结构这一做法。
----

### BPE
> 1. 字节对编码(Byte pair encoder)，也称digram coding双字母组合编码，主要目的是为了数据压缩，算法描述为字符串里频率最常见的一对字符被一个没有在这个字符串中出现的字符代替并进行迭代过程。
> 2. BPE在欧洲语系中表现的有效一些，因为欧洲语系存在词缀等概念。在汉语翻译中，引入BPE会带来很多诡异的字符。
> 3. 在nlp中，sequence-to-sequence模型中使用较小的词表有助于提高系统性能。
> 4. 解决OOV和Rare word问题。

+ [算法过程](https://en.wikipedia.org/wiki/Byte_pair_encoding)
	+ **编码过程**
		+ 假如对字符串"aaabdaaabac"进行编码，其中字符对"aa"(以2个字符共现为例)出现次数最高，那么使用字符'Z'来替代"aa": "ZabdZabac"('Z'="aa")；接下来，字符对"ab"出现次数最高，同样，使用字符'Y'来替代"ab"："ZYdZYac"('Y'="ab"，'Z'="aa")，同样，字符对"ZY"出现次数最高，用字符'X'替代"ZY"："XdXac"('X'="ZY", 'Y'="ab"，'Z'="aa")。最后，连续字符对出现的次数都为1，编码结束，得到codec文件。codec文件保存的是编码过程中的字符对，文件中最开始的是最先保存的字符，具有较高的优先级。
	+ **解码过程**
		+ 解码过程按照相反的顺序进行替换，即首先将词拆成一个一个的字符，然后按照codec文件中的字符对进行合并。

+ **[官方代码](https://github.com/rsennrich/subword-nmt)**

+ **[别人解释](https://blog.csdn.net/u013453936/article/details/80878412)**

----

### SentencePiece
> SentencePiece为字词的切分算法，在英文中就是BPE。

+ **算法过程**
	+ 拆分句子中有两个变量，一个为词表和句子的切分序列。EM算法，句子的切分序列为隐变量。开始时，随机初始化一个词表和随机切分一下句子。
		+ 1. 固定词表，求一个句子困惑度最低的切分序列。
		+ 2. 根据这个切分序列求固定词表，剔除一个词，然后计算困惑度，最后对困惑度设定一个阈值，筛选一些对语料集影响较大的词。



+ **[官方代码](https://github.com/google/sentencepiece)**


+ **[别人解释](https://blog.csdn.net/eqiang8848/article/details/88548915)**

----

### Attention的实现
> 苏剑林. (2018, Jan 06). 《《Attention is All You Need》浅读（简介+代码） 》[Blog post]. Retrieved from https://kexue.fm/archives/4765

+ keras实现
	```python
	from keras import backend as K
	from keras.engine.topology import Layer

	class Position_Embedding(Layer):
		def __init__(self, size=None, mode='sum', **kwargs):
			self.size = size #必须为偶数
			self.mode = mode
			super(Position_Embedding, self).__init__(**kwargs)
        
		def call(self, x):
			if (self.size == None) or (self.mode == 'sum'):
				self.size = int(x.shape[-1])
            
			batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
			position_j = 1. / K.pow(10000., 2*K.arange(self.size/2, dtype='float32')/self.size) # shape: (self.size/2,)
			position_j = K.expand_dims(position_j, 0)  # shape: (1, self.size/2)
			position_i = K.cumsum(K.ones_like(x[:,:,0]), 1)-1  # K.arange不支持变长，只好用这种方法生成，shape: (batch_size, seq_len)
			position_i = K.expand_dims(position_i, 2)  # shape: (batch_size, seq_len, 1)
			position_ij = K.dot(position_i, position_j)  # shape: (batch_size, seq_len, 1) x (1, self.size/2)  —> (batch_size, seq_len, self.size/2)
			position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
			if self.mode == 'sum':
				return position_ij + x
			elif self.mode ==  'concat':
				return K.concatenate([position_ij, x], 2)
    
		def compute_output_shape(self, input_shape):
			if self.mode == 'sum':
				return input_shape
			elif self.mode == 'concate':
				return (input_shape[0], input_shape[1], input_shape[2]+self.size)
        
	class Attention(Layer):
		def __init__(self, nb_head, size_per_head, mask_right=False, **kwargs):
			self.nb_head = nb_head
			self.size_per_head = size_per_head
			self.output_dim = nb_head*size_per_head
			self.mask_right = mask_right
			super(Attention, self).__init__(**kwargs)
			
		def build(self, input_shape):
			self.WQ = self.add_weight(name='WQ',shape=(input_shape[0][-1], self.output_dim), initializer='glorot_uniform', trainable=True)
			self.WK = self.add_weight(name='WK', shape=(input_shape[1][-1], self.output_dim), initializer='glorot_uniform', trainable=True)
			self.WV = self.add_weight(name='WV', shape=(input_shape[2][-1], self.output_dim), initializer='glorot_uniform', trainable = True)
			super(Attention, self).build(input_shape)
			
		def Mask(self, inputs, seq_len, mode='mul'):
			if seq_len == None:
				return inputs
			else:
				mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
				mask = 1 - K.cumsum(mask, 1)
				for _ in range(len(inputs.shape)-2):
					mask = K.expand_dims(mask, 2)
				if mode == 'mul':
					return inputs * mask
				if mode == 'add':
					return inputs - (1-mask)*1e12
				
		def call(self, x):
			# 如果只传入Q_seq, K_seq, V_seq，那么就不做Mask
			# 如果只传入Q_seq, K_seq, V_seq, Q_len, V_len，那么对多余部分做Mask
			if len(x) == 3:
				Q_seq, K_seq, V_seq = x
				Q_len, V_len = None, None
			elif len(x) == 5:
				Q_seq, K_seq, V_seq, Q_len, V_len = x
			
			# 对Q、K、V做线性变换
			Q_seq = K.dot(Q_seq, self.WQ)
			Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
			Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
			K_seq = K.dot(K_seq, self.WK)
			K_seq = K.reshape(K_seq, (-1,K.shape(K_seq)[1], self.nb_head, self.size_per_head))
			K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
			V_seq = K.dot(V_seq, self.WV)
			V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
			V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
			#计算内积，然后mask，然后softmax
			A = K.batch_dot(Q_seq, K_seq, axes=[3,3])/self.size_per_head**0.5
			A = K.permute_dimensions(A, (0,3,2,1))
			A = self.Mask(A, V_len, 'add')
			A = K.permute_dimensions(A, (0,3,2,1))
			if self.mask_right:
				ones = K.ones_like(A[:1, :1])
				mask = (ones - K.tf.matrix_band_part(ones, -1, 0))*1e12
				A = A - mask
			A = K.softmax(A)
			# 输出并mask
			O_seq = K.batch_dot(A, V_seq, axes=[3,2])
			O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
			O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
			O_seq = self.Mask(O_seq, Q_len, 'mul')
			return O_seq
		
		def compute_output_shape(self, input_shape):
			return (input_shape[0][0], input_shape[0][1], self.output_dim)
	
	```
+ tensorflow实现
	```python
	import tensorflow as tf

	'''
	inputs是一个形如(batch_size, seq_len, word_size)的张量；
	函数返回一个形如(batch_size, seq_len, position_size)的位置张量。
	'''
	def Position_Embedding(inputs, position_size):
		batch_size,seq_len = tf.shape(inputs)[0],tf.shape(inputs)[1]
		position_j = 1. / tf.pow(10000., \
								 2 * tf.range(position_size / 2, dtype=tf.float32 \
								) / position_size)
		position_j = tf.expand_dims(position_j, 0)
		position_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
		position_i = tf.expand_dims(position_i, 1)
		position_ij = tf.matmul(position_i, position_j)
		position_ij = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 1)
		position_embedding = tf.expand_dims(position_ij, 0) \
							 + tf.zeros((batch_size, seq_len, position_size))
		return position_embedding


	'''
	inputs是一个二阶以上的张量，代表输入序列，比如形如(batch_size, seq_len, input_size)的张量；
	seq_len是一个形如(batch_size,)的张量，代表每个序列的实际长度，多出部分都被忽略；
	mode分为mul和add，mul是指把多出部分全部置零，一般用于全连接层之前；
	add是指把多出部分全部减去一个大的常数，一般用于softmax之前。
	'''
	def Mask(inputs, seq_len, mode='mul'):
		if seq_len == None:
			return inputs
		else:
			mask = tf.cast(tf.sequence_mask(seq_len), tf.float32)
			for _ in range(len(inputs.shape)-2):
				mask = tf.expand_dims(mask, 2)
			if mode == 'mul':
				return inputs * mask
			if mode == 'add':
				return inputs - (1 - mask) * 1e12

	'''
	普通的全连接
	inputs是一个二阶或二阶以上的张量，即形如(batch_size,...,input_size)。
	只对最后一个维度做矩阵乘法，即输出一个形如(batch_size,...,ouput_size)的张量。
	'''
	def Dense(inputs, ouput_size, bias=True, seq_len=None):
		input_size = int(inputs.shape[-1])
		W = tf.Variable(tf.random_uniform([input_size, ouput_size], -0.05, 0.05))
		if bias:
			b = tf.Variable(tf.random_uniform([ouput_size], -0.05, 0.05))
		else:
			b = 0
		outputs = tf.matmul(tf.reshape(inputs, (-1, input_size)), W) + b
		outputs = tf.reshape(outputs, \
							 tf.concat([tf.shape(inputs)[:-1], [ouput_size]], 0)
							)
		if seq_len != None:
			outputs = Mask(outputs, seq_len, 'mul')
		return outputs

	'''
	Multi-Head Attention的实现
	'''
	def Attention(Q, K, V, nb_head, size_per_head, Q_len=None, V_len=None):
		#对Q、K、V分别作线性映射
		Q = Dense(Q, nb_head * size_per_head, False)
		Q = tf.reshape(Q, (-1, tf.shape(Q)[1], nb_head, size_per_head))
		Q = tf.transpose(Q, [0, 2, 1, 3])
		K = Dense(K, nb_head * size_per_head, False)
		K = tf.reshape(K, (-1, tf.shape(K)[1], nb_head, size_per_head))
		K = tf.transpose(K, [0, 2, 1, 3])
		V = Dense(V, nb_head * size_per_head, False)
		V = tf.reshape(V, (-1, tf.shape(V)[1], nb_head, size_per_head))
		V = tf.transpose(V, [0, 2, 1, 3])
		#计算内积，然后mask，然后softmax
		A = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(size_per_head))
		A = tf.transpose(A, [0, 3, 2, 1])
		A = Mask(A, V_len, mode='add')
		A = tf.transpose(A, [0, 3, 2, 1])
		A = tf.nn.softmax(A)
		#输出并mask
		O = tf.matmul(A, V)
		O = tf.transpose(O, [0, 2, 1, 3])
		O = tf.reshape(O, (-1, tf.shape(O)[1], nb_head * size_per_head))
		O = Mask(O, Q_len, 'mul')
		return O
	
	
	```
----

### BeamSearch
> 在seq2seq中，decoder是“单向递归”的，其解码过程是递归的（利用预测得到的t时刻的结果去预测t+1时刻的结果）。这种递归解码方式存在一种问题：假如t时刻的结果预测错误，那么会导致后面的训练和预测都无意义。因此，在训练过程中引入一种解码搜索方法——Beam Search。

![seq2seq解码过程](https://i.loli.net/2019/06/17/5d075c5ccb14317367.jpg)

> seq2seq的解码过程如上图的概率公式所示，每次递归过程都是选择当前最大概率的输出（贪心搜索），这种搜索方案的结果未必最优。seq2seq使用beam search的折中方案。

+ 思想：在每次计算前，只保留当前最优的top k个候选结果。比如：当k=3时，第一步选择使得概率最大的前3个值，然后分别递归计算下一时刻的每个值的概率，然后各取前3个值，这样的话就是9种组合了，然后计算每一种组合的总概率，选择最大的前3个，依次递归直至出现<end>为止。**k=1时为贪心搜索**

----