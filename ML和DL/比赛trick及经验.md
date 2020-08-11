数据增强方法——增强泛化性能
1、mix-up，配合dropout使用才会有一定效果
2、label-smoothing，拟合更快，效果会优于mix-up+dropout，但参数要合适
3、disturb-Label
4、pseduo-label
5、cutMix
6、cutout
7、label-embedding
8、Penalizing low entropy




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

