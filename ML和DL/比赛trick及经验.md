数据增强方法——增强泛化性能
1、mix-up，配合dropout使用才会有一定效果
    ## 使用相同标签数据扩增
    ## 在全量数据上数据扩增
    
    ## mixup
    def dataAugment_mixup(x,y,batch_size=32,alpha=0.2):
        '''
        x: training_data, dim=4, shape=[batch,width,height,channel]
        y: one-hot label, dim=2, shape=[batch,class]
        alpha: hyper-parameter, default is 0.2
        '''
        from random import shuffle
        length = x.shape[0]
        arr = [i for i in range(length)]
        steps = length // batch + 1
        while True:
            shuffle(arr)
            for i in range(steps):

                if alpha > 0.:
                    if i == steps-1:
                        weight = np.random.beta(alpha,alpha,length-batch*i)
                        x_weight = weight.reshape(length-batch*i,1,1,1)
                        y_weight = weight.reshape(length-batch*i,1)
                        index = np.random.randint(steps)
                        yield (x[arr[i*batch:(i+1)*batch]]*x_weight + x[arr[index*(length-batch*i):(index+1)*(length-batch*i)]]*(1-x_weight), y[arr[i*batch:(i+1)*batch]]*y_weight+y[arr[index*(length-batch*i):(index+1)*(length-batch*i)]]*(1-y_weight))

                    else:
                        weight = np.random.beta(alpha, alpha, batch)
                        x_weight = weight.reshape(batch,1,1,1)
                        y_weight = weight.reshape(batch,1)
                        index = np.random.randint(steps)
                        yield (x[arr[i*batch:(i+1)*batch]]*x_weight + x[arr[index*batch:(index+1)*batch]]*(1-x_weight), y[arr[i*batch:(i+1)*batch]]*y_weight+y[arr[index*batch:(index+1)*batch]]*(1-y_weight))

                else:
                    yield (x[arr[i*batch:(i+1)*batch]], y[arr[i*batch:(i+1)*batch]])
        
SMOTE算法：x + rand(0,1)*(x_1 - x)，这里x是原始样本，x_1是x的近邻样本    
    
2、label-smoothing，拟合更快，效果会优于mix-up+dropout，但参数要合适
    '''
    def ls_loss(y_true, y_pred, e=0.1):
        loss1 = tf.keras.backend.categorical_crossentropy(y_true, y_pred)
        loss2 = tf.keras.backend.categorical_crossentropy(tf.keras.backend.ones_like(y_pred)/num_classes, y_pred)
        return (1-e)*loss1 + e*loss2
    '''
3、disturb-Label
4、pseduo-label
5、cutMix
6、cutout
7、label-embedding
    ## label smoothing
    def data_enhanced1(x,y,epsilon=0.1):
        v = y.shape[-1]

        y = (1-epsilon)*y + (epsilon/v)
        return x,y


8、Penalizing low entropy

9、针对时序数据的数据扩增
   -- https://stats.stackexchange.com/questions/320952/data-augmentation-strategies-for-time-series-forecasting
   方法1：
    (1) 时序数据A[i](1<=i<=n)，假定0<e<|A[i+1]-A[i]|，其中(1<=i<=n)；
    (2) 构建新数据B[i] = A[i]+r[i]，r[i]属于分布N(0,e/2)；
   方法2：
    (1) 



backbone基础
1、ResNet的bottleneck block



损失函数修改
1、长尾问题损失函数修改，直接改信息熵



修改cnn提高性能可以用以下方法： 

designing deeper network structures [30]（VGG）[34]（Inception）,

exploring or learning non-linear activation functions [5]（Maxout）[21]（NIN）[7]（Leaky ReLU),

developing new pooling operations [42](Stochastic Pooling)[6](Fractional Max-Pooling)[18](Generalizing PoolingFunctions in Convolutional Neural Networks: Mixed, Gated,and Tree), 

introducing better optimization techniques [19](DeeplySupervisedNets),

regularization techniques preventing the network from over-ﬁtting [8](Dropout)[38](DropConnect)

【15】cross-map normalization 

Local Contrast Normalization(LCN) for data preprocessing [5].


BN层在卷积层和激活层中间
使用全局池化会优于使用FC层，一般前面的网络使用最大池化，最后一层使用平均池化


学习率
(1) step learning rate decay
    随着epoch增大，学习率不断减去一个小的数值；
(2) cosine learning rate decay
    lr = 1/2*(1+cos(t*pi/T))*lr_s  lr_s为初始学习率，T为总steps, t为第t个batch;


在训练集训练后，在验证集继续训练两个 epochs（小技巧，可能很有用）

均值编码：一种处理高基数(取值较多)类别特征的简单高效方法；但容易过拟合，需配合使用正则化；见https://www.kesci.com/mw/project/5be7a3b4954d6e00106320e5

对于高度偏斜分布的数据，需要利用log(1+原始值)进行处理，处理后会接近高斯分布；但是提交预测值时，需要进行转换：predictions=np.exp(log_predictions)  


# 直播——阿水
+ 特征处理常规步骤
    + 步骤1：对字段提取&编码
        + 1）类别特征；数值特征；文本特征；图像特征
    + 步骤2：对字段统计编码
        + count encoding：对列统计
        + target counting
    + 步骤3：对字段分组统计
        + 按target进行分组统计；按字段进行分组统计
    + 步骤4：构建高阶交叉特征
        + 二阶交叉
        + 三阶交叉
    + 步骤4：特征子集筛选
+ 对于线上分布线下分布不一致问题
    + 原因：可能人为地调整了线上和线下数据的分布比例
        + 自己的理解：可以通过看看测试评分里是否有对不同类别的权重，然后和训练数据里不同类别的的权重对比；
    + 第一，查看是否过拟合
    + 第二，对训练集和测试集进行分类判断，剔除特征重要性高的特征
+ 数据适用场景
    + 一般来说，结构化数据适合传统机器学习模型；非结构化特征适合深度学习；
  