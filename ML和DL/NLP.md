## NLP

- [content]
	- [TFIDF](#TF-IDF)
	- [word2vec](#word2vec)
	- [Glove](#Glove)
	- [Fasttext](#Fasttext)
	- [为什么有了word2vec，还要Glove、ELMO等？](#为什么有了word2vec，还要Glove、ELMO等？)
	- [BPE](#BPE)
	- [SentencePiece](#SentencePiece)
	- [Attention](#Attention)
	- [Beam Search](#BeamSearch)
    - [transformer](#transformer)
    - [bert](#bert)
    - [XLNet](#XLNet)
    - [Albert——降低参数量、减少内存、提升模型效果](#Albert——降低参数量、减少内存、提升模型效果)
    - [RoBERTa](#RoBERTa)
    - [WordPiece](#WordPiece)
    - [BatchNorm和LayerNorm](#BatchNorm和LayerNorm)
    - [NER](#NER)
	
	
### TF-IDF
+ TF：term frequency，词频，表示词在文本中出现的频率；
+ IDF：Inverse Document Frequency，逆向文件频率，为总文件数目除以包含该词的文件数量；
    + 公式：log(语料库中文档总数/(包含该词的文件数量+1))
    + **IDF为什么用对数？**
        + 涉及到信息论的知识，指IDF相当于一个权重，当该词越容易分辨，则相应权重也会特别大。
+ TF-IDF
    + 公式：TF*IDF
+ 应用
    + 搜索引擎
    + 关键词提取
    + 文本相似性
    + 文本摘要
    
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

### Attention
+ self-attention
    + 
+ hard-attention
    + 直接定位到某个key并将其概率设为1，其它key的概率为0——每个时刻模型的序列只有一个取值为1，其余为0
    + 优点
        + 计算量小
    + 缺点
        + 对对齐方式要求很高
        + 不可导，只能通过强化学习训练
+ soft-attention (global-attention)
    + 利用所有key并计算概率权重，最后加权求和
    + 优点
        + 考虑了所有信息，计算量大;
    + 缺点
        + 计算量大
+ local-attention
    + 对一个窗口区域进行计算，综合soft-attention和hard-attention实现，先用hard-attention定位到某个点，以这个点为中心区域得到一个区域，并在这个区域进行soft-attention。

+ attention实现
    + 苏剑林. (2018, Jan 06). 《《Attention is All You Need》浅读（简介+代码） 》[Blog post]. Retrieved from https://kexue.fm/archives/4765
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

### transformer
+ attention中scaled的作用
    + 当Q和K维度很大时，点积结果会很大，不经过scaled处理，使得softmax会将元素最大的位置非常接近1(类似one-hot)，导致反向传播求导时梯度为0(梯度消失)。
+ 绝对位置编码
    + 公式：(U_i + E_i)*(W_q)*(W_k)*(U_j+E_j) = (E_i*W_q*W_k*E_j)+(E_i*W_q*W_k*U_j)+(U_i*W_q*W_k*E_j)+(U_i*W_q*W_k*U_j)
+ self-attention
    + 核心：用文本中其它词来增强目标词的语义表示，更好地利用上下文信息；
    + 公式：softmax(Q*K/sqrt(d))*V
    + attention中Q、K、V为什么要先乘以权重？
        + 通过乘以不同权重，能对随机的embedding先做一个拟合，获得相对来说更好的表示；
        + 通过乘以不同权重使得Q、K、V都不一样，能更好地利用上下文信息；
    + attention那块Q、K、V能全部一样吗？
        + 不能，由于Q,K是用来计算权重的，如果Q,K都一样的话，Q与K点积后，肯定是自身值最大，导致softmax后权重最大的都是自身；
    + 如果attention中Q、K乘以权重，V不乘权重可以吗？
        + 理论上是可以的，不过V乘权重后能获得更好的表示而不是随机值；
    + attention为什么做放缩？
        + 能将方差控制为1，有效控制梯度消失的问题；
    + 为啥使用multi-head?
        + multi-head可以注意到不同子空间的信息，捕捉更加丰富的特征信息；
+ transform什么地方做了权重共享？
    + encoder和decoder见的embedding层权重共享；
    + decoder中embedding层和fc层权重共享；

### transformer-xl
+ 段级循环：将长文本分成segment，下一segment会利用缓存的前一segment的隐向量（缓存的隐向量只参与前向传播，不参与后向传播）
    + 具体做法：
        + attention处：query维度为[segment_length, hidden_size]、key维度为[2*segment_length, hidden_size]、value维度为[2*segment_length, hidden_size]
+ 相对位置编码
    + 公式：(E_i*W_q*W_k*E_j)+(E_i*W_q*W_k*R_(i-j))+(u*W_k*E_j)+(v*W_k*R_(i-j))
        + 将所有U_j修改为R_(i-j)，表示对key来说将绝对位置转换为相对query(i)的位置；
        + U_i表示query相关的绝对位置向量，改为相对位置后，query就和自己的位置无关，所以将U_i*W_q用与位置无关的u和v代替，u和v需要训练；
    + 相对位置的窗口小于等于4

### bert
+ 只用了transformer-encoder结构：Multi-Attention + FFN
    + BERT_base: L=12，H=768，A=12，total parameter = 110M  (其中，A是自注意力头的个数)
    + BERT_large：L=24，H=1024，A=16，total_parameter = 340M
+ 随机mask：只在准备数据时做mask，相当于只mask一次；并且会替换原有单词的一个部分，比如: phi ##am ##mon ---> phi [MASK] ##mon
+ next sentence predict(NSP)：正例是文章中连续的两个句子，而负例则是从两篇文档中各选一个句子构造而成；
+ character-level BPE
+ Gelu激活函数：由于Relu激活函数缺乏随机因素而改进——其实是dropout、zoneout、Relu的结合
    + 公式：Gelu(x) = x·P(X<=x)，P符从正态分布
        + 近似计算：Gelu(x) = 0.5·x·(1+tanh(sqrt(2/pi)·(x+0.044715·x·x·x)))
+ bert中的优化器:
+ 由于采用的是transformer结构，能并发执行;
+ 缺点：
    + 训练数据和测试数据之间不一致性(Discrephancy)：训练时会随机mask，测试时不会造成的；
        + 缓解策略：修改为每批次只对15%的训练数据处理，并且80%概率将字符用[MASK]标记，10%概率随机替换，10%概率保持不变；
    + 由于每批次只对15%的标记进行预测，导致模型需要更多的训练步骤；
    + 当多个连续的TOKENS都被[MASK]标记时，BERT会假设这些单词之间是独立的；
    + 由于是Denoising auto-encoder结构，无法像NNLM模型做句子或文本生成；
    + 对长文本任务效果较不明显；
+ 改进方向
    + 针对MASK时只对单词的部分字符进行MASK，可以采用whole word masking技术优化，比如：phi ##am ##mon ---> [MASK][MASK][MASK]
    + 针对静态MASK，可以采用动态MASK方法；
+ 输入
    + 3种词向量
        + 词的embedding；
        + 位置的embedding；
        + segment的embedding；
            + 为了将多个句子区分：第一个句子全为0，第二个句子全为1，...
            + 只有一个句子时，全用0表示；
    + 句子开头有特殊符[CLS]，句子结尾有特殊符[SEP]；
    + 如果两个句子同时输入，则只有开头有[CLS]，后面的句子没有[CLS]，句子结尾符只有后一个句子有[SEP]；
    + **句子开头的[CLS]本身无任何意义，但是它能表示整个句子的语义**
+ 问题
    + bert如何处理梯度消失和梯度爆炸？
        + short_cut
        + layerNorm
        + attention中Q和K的点积会除以维度
        + 激活函数

### XLNet
+ 排列语言模型
    + 给出所有组合：比如序列[1,2,3,4]，组合有24种
        + 假如预测目标是3，
    + 实现思路：双流注意力机制，在transformer中attention层实现mask，对一个句子的末尾k个词进行mask
    
+ 采用相对位置编码和transformer-xl结构
    + 对长文本任务效果有较好提升

+ 采用更大的数据量

+ 相比bert
    + 采用auto-regression模型，可以完成生成任务
    + 解决mask的负面影响：训练时有mask，测试时没有mask
    + 双流注意力机制
    + 引入transformer-xl

+ **https://zhuanlan.zhihu.com/p/70257427**

### Albert——降低参数量、减少内存、提升模型效果
+ 加大隐层维度，并将词向量分解：bert中原始词向量大小为V×E(E==H)，由于词嵌入学习的是与上下文无关的表示，隐层学习的是与上下文相关的表示，后者更加复杂，所以需要增加隐层的大小；
由于E==H，增加隐层大小又会带来词向量空间太大，因此打破E和H的关系，将V×H分解为V×E和E×H；
+ 多层共享参数：1）共享feed-forward network参数；2）共享attention参数；3）共享全部参数；
+ 提出Sentence-order prediction (SOP)来取代NSP：1）NSP这种方式由于两篇文档的话题通常不同，模型会更多的通过话题去分析两个句子的关系，而不是句子间的连贯性；2）其正例与NSP相同，但负例是通过选择一篇文档中的两个连续的句子并将它们的顺序交换构造的。
+ 增加模型的层数或隐层大小确实能够在一定程度上提升模型的表现。但当大小增加到一定量时，反而会使模型表现变差。
+ 加入额外数据能提升模型表现；由于没有过拟合，去除DropOut后表现更好；
+ warm-start，训练深层网络(12层)时，可以先训练浅层网络(6层)，再在基础上做微调，这样能加快收敛——为啥考虑6层:可能考虑到全局参数共享吧；
+ 缺点：
    + 模型推理时间并没有减少，因为层数还是这么多；（减少层数能减少推理时间——贝壳找房：《ALBERT在房产领域的实践》）

### RoBERTa
+ 训练时间更长
+ batch_size加大，从BERT的256增大到了2K甚至8K
+ 训练数据加大
+ 移除NSP损失
+ 训练序列更长：数据连续的从一个或多个文档上获取直至512，文档间采用标识符分开
+ static mask：不同于bert中mask机制，1）首先，数据复制dupe_factor份；2）同一条数据用不同的mask，并且用于不同的epoch；
+ dynamic mask：动态调整MASK机制，喂给模型时随机mask
+ byte-level BPE：
+ 修改优化器，将优化器中二阶矩估计的超参数beta_2从0.999减小到0.98，并将防止分母为0的超参数epsilon从1e-8增大到1e-6；

### WordPiece
## Byte-Pair Encoding(BPE)：字节对编码
+ 第一步，构建词典、统计词频：{词：词频}，
+ 第二步，对词按字母切分，然后添加</w>表示结尾，
+ 第三步，找出出现次数最多的两个字符进行合并
+ 重复第三步直至达到设置频率阈值
    ```
    原始词表 {'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3, 'l o w </w>': 5}
    出现最频繁的序列 ('s', 't') 9
    合并最频繁的序列后的词表 {'n e w e st </w>': 6, 'l o w e r </w>': 2, 'w i d e st </w>': 3, 'l o w </w>': 5}
    出现最频繁的序列 ('e', 'st') 9
    合并最频繁的序列后的词表 {'l o w e r </w>': 2, 'l o w </w>': 5, 'w i d est </w>': 3, 'n e w est </w>': 6}
    出现最频繁的序列 ('est', '</w>') 9
    合并最频繁的序列后的词表 {'w i d est</w>': 3, 'l o w e r </w>': 2, 'n e w est</w>': 6, 'l o w </w>': 5}
    出现最频繁的序列 ('l', 'o') 7
    合并最频繁的序列后的词表 {'w i d est</w>': 3, 'lo w e r </w>': 2, 'n e w est</w>': 6, 'lo w </w>': 5}
    出现最频繁的序列 ('lo', 'w') 7
    合并最频繁的序列后的词表 {'w i d est</w>': 3, 'low e r </w>': 2, 'n e w est</w>': 6, 'low </w>': 5}
    出现最频繁的序列 ('n', 'e') 6
    合并最频繁的序列后的词表 {'w i d est</w>': 3, 'low e r </w>': 2, 'ne w est</w>': 6, 'low </w>': 5}
    出现最频繁的序列 ('w', 'est</w>') 6
    合并最频繁的序列后的词表 {'w i d est</w>': 3, 'low e r </w>': 2, 'ne west</w>': 6, 'low </w>': 5}
    出现最频繁的序列 ('ne', 'west</w>') 6
    合并最频繁的序列后的词表 {'w i d est</w>': 3, 'low e r </w>': 2, 'newest</w>': 6, 'low </w>': 5}
    出现最频繁的序列 ('low', '</w>') 5
    合并最频繁的序列后的词表 {'w i d est</w>': 3, 'low e r </w>': 2, 'newest</w>': 6, 'low</w>': 5}
    出现最频繁的序列 ('i', 'd') 3
    合并最频繁的序列后的词表 {'w id est</w>': 3, 'newest</w>': 6, 'low</w>': 5, 'low e r </w>': 2}
    ```

### BatchNorm和LayerNorm
+ LayerNorm
    + transformer中使用LayerNorm：主要是利用正态分布公式对词向量进行归一化，让模型更好的学习特征分布，更快收敛
+ BatchNorm
    + BatchNorm主要是对同一batch中的同一维度进行归一化，即对[batch_size, seq_size, embed]中[batch_size, seq_size]进行归一化
+ 为什么nlp中不适用BatchNorm？
    + 一个batch中句子长度不一样，反而会增大方差
    + 一个batch中的token关联性不大
    + BatchNorm会使得Bacth的均值和方差不稳定
    + LayerNorm能够对self-attention处理得到的向量进行归一化，降低方差，加速收敛
    

### NER
> 命名实体识别：named entity recognize，从文本中识别出命名性指称项，为关系抽取等任务做铺垫；
> 任务：识别出人名PER、地名LOC、组织机构名ORG；（时间、货币名称等实体类型可以用正则等识别）

+ 传统方法
    + 基于规则的方法：例如，对于中文来说，“说”、“老师”等词语可作为人名的下文，“大学”、“医院”等词语可作为组织机构名的结尾，还可以利用到词性、句法信息。
    + 基于特征模板的方法：将NER视作序列标注任务，利用已训练模型对句子的各个位置进行标注；方法：特征模板+常用模型：HMM、CRF等
        + 特征模板+CRF：特征模板是人工定义的二值特征函数，试图挖掘命名实体内部以及上下文的构成特点。对于句子中的给定位置来说，提特征的位置是一个窗口，即上下文位置。不同的特征模板之间可以组合形成一个新的特征模板。CRF的优点在于其为一个位置标注的过程中可以利用到此前已经标注的信息，利用Viterbi解码得到最优序列。对句子中的各个位置提取特征时，满足条件的特征取值为1，不满足条件的特征取值为0；然后把特征喂给CRF，training阶段建模标签的转移，进而在inference阶段为测试句子的各个位置做标注。
    + 基于神经网络的方法：RNN+CRF
        + 输入X：[x_1,...,x_n]
        + RNN
            + 输入：[batch, seq_length] ===> [batch, seq_length, seq_embed]
            + 输出P：[batch, seq_length, label_num]
        + CRF：[batch, seq_length, label_num] + [label_num+2, label_num+2] ===> [batch, label_num]
            + 参数矩阵A：[label_num+2, label_num+2]，A_ij表示从第i个标签到第j个标签的转移得分
            + 加2的原因：句子首部添加起始状态，尾部添加终止状态；
        + 输出Y：[y_1,...,y_n]
        + score函数：一部分是RNN输出的P决定，一部分是由CRF的转移矩阵A
            + 对于一个句子，x为该句子序列，y为该句子的标注序列，p为RNN输出，A为转移矩阵
                + score(x,y) = sum_i(1,n)(p_(i,y(i))) + sum_i(1,n+1)(A_(y(i-1)_y(i)))
        + 输出：P(y|x) = exp(score(x,y))/(sum_y'(exp(score(x,y'))))
        + 缺点：对每个token打标签是独立的分类，无法直接利用上文已经预测的标签，导致最后预测的标签序列是不合理的，例如：B-PER后面是不可能跟I-LOC。

+ 标注集
    + BIO标注集
        + B-PER、I-PER代表人名首字、人名非首字
        + B-LOC、I-LOC代表地名首字、地名非首字
        + B-ORG、I-ORG代表组织机构名首字、组织机构名非首字
        + O代表该字不属于命名实体的一部分