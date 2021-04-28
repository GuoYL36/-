
## Style-based GANs-Generating and Tuning Realistic Artifical Faces
> [Style-based GANs](https://www.lyrn.ai/2018/12/26/a-style-based-generator-architecture-for-generative-adversarial-networks/)


![ProGAN overview](https://github.com/GuoYL36/other/tree/master/paper/img/ProGAN-chart-1.png)

> ProGAN生成高质量图片，但是和很多模型一样，控制生成图片的特定特征是有限的。换句话说，特征是纠缠的，如果对输入稍稍调整，就会影响多个特征。打个比方，现实中只要改变一个基因就会影响多个特征。

![ProGAN progressive training from low to high res layers](https://github.com/GuoYL36/other/tree/master/paper/img/ProGAN.gif)


### How StyleGAN works
> StyleGAN文章提供了ProGAN的图片生成器的更新版本，主要关注生成器网络。文中作者发现ProGAN中progressive layer的潜在优势是如果能得到合理利用，那么就能够控制图片的不同视觉特征。层数越少(分辨率越低)，它影响越Coarser(粗糙的)特征。这篇文章将特征分成三类：1、Coarse(粗糙)——分辨率达到8×8——影响姿态、general头发类型、脸部形状等；2、Middle——分辨率为16×16到32×32——影响好的脸部特征、头发类型、睁眼或闭眼等；3、Fine——分辨率为64×64到1024×1024——影响颜色方案(眼睛、头发和皮肤)和微特征。



### Mapping Network
> Mapping Network的目的是将输入向量编码成中间向量，利用这个中间向量的不同元素控制着不同的视觉特征。这是一个non-trivial过程，因为直接控制输入向量的视觉特征是受限的，它必须服从训练数据的概率密度。<br>例如，如果数据集中黑头发人物图片更普遍，那么更多的输入值将被映射成黑头发的特征。结果，模型不能够将部分输入(向量元素)映射成特征，这个现象称作为特征纠缠。然而，通过使用另一神经网络，模型能够生成一个不需要服从训练数据分布的向量，这个向量能够减少特征之间的相关性。<br>Mapping Network由8个全连接层组成，其输出w的维度和输入维度(512×1)一致。

![The generator with the Mapping Network(in addition to the ProGAN synthesis network)](https://github.com/GuoYL36/other/tree/master/paper/img/StyleGAN-generator-Mapping-network.png)

### Style Modules(AdaIN)
> [AdaIN](https://arxiv.org/abs/1703.06868)(Adaptive Instance Normalization)模块将编码信息w(来自Mapping-network)转变为已生成图片。这个模块被增加到Synthesis网络的每一个分辨率层，并在那一层定义特征的视觉表示：1、卷积层输出的每一个channel首先进行归一化，确保步骤3中的scaling和shifting有期望的效应；2、使用另一个全连接层(标记为 A )将中间向量w变换为每一个channel的scale和bias；3、scale和bias向量调整卷积输出的每个channel，可以认为是卷积的每个滤波器的权重。这个调整可以将来自w的信息翻译成视觉表示。

![The generator's Adaptive Instance Normalization (AdaIN)](https://github.com/GuoYL36/other/tree/master/paper/img/StyleGAN-generator-AdaIN.png)

### Removing traditional input
> 很多模型，包括ProGAN都使用随机输入创建生成器的初始图片(相当于4×4的水平)。StyleGAN团队发现图片特征由w和AdaIN控制，因此初始输入会被省略和由常数值替代。虽然文章中没有解释为什么能提升性能，但是一个假设是它能够减少特征纠缠——对于网络来说，它更容易仅仅使用不依赖于纠缠的输入向量的w去学习。

![The Synthesis Network input is replaced with a constant input](https://github.com/GuoYL36/other/tree/master/paper/img/StyleGAN-generator-Input.png)


### Stochastic variation
> 人类脸上有很多方面是微小的并且能被随机地观察到，比如：雀斑、头发的准确放置、皱纹，这些特征使得图片更加真实和增加输出的多样性。将这些小特征插入GAN图片的常用方法是将随机噪声加入输入向量中。然而，在许多例子中，由于特征纠缠现象会导致图片的其他特征受影响，所以控制噪声效应是很复杂的。StyleGAN中的噪声以相同的方式被加到AdaIN机制——一种Scaled噪声被加入AdaIN模块之前的每一个channel中，并改变了一些分辨率层中特征的视觉表示。

![Adding scaled noise to each resolution level of the synthesis network](https://github.com/GuoYL36/other/tree/master/paper/img/StyleGAN-generator-Noise.png)

### Style mixing
> StyleGAN生成器使用了Synthesis Network中每一层的中间向量，这可能导致网络学习到的levels是相关的。为了减少相关性，模型随机选择两个输入向量并为它们生成中间向量w。然后它对第一个中一些levels进行训练并且（在一个随机点上）切换到另一个其它以训练其余的levels。随机切换确保网络将不会学习并依赖于levels之间的相关性。虽然在所有数据集上，它没有改善模型性能，但这个概念有一个非常有趣的副作用 - 它能够以连贯的方式组合多个图像（如下面的视频所示）。该模型生成两个图像A和B，然后通过从A中获取低level特征并从B中获取其余特征来组合它们。

### Truncation trick in W
> 生成模型面临的挑战之一是处理训练数据中表现不佳的领域。生成器无法学习它们并创建类似于它们的图像（而是创建看起来很糟糕的图像）。为了避免生成不良图像，StyleGAN截断中间向量w，强制它接近“平均”中间向量。训练模型后，通过选择许多随机输入产生“平均”w_avg; 用Mapping Network生成它们的中间向量; 并计算这些向量的平均值。在生成新图像时，不是直接使用Mapping Network输出，而是将w转换为w_new = w_avg +\Ψ（w-w_avg），其中\Ψ的值定义图像距“平均”图像的距离（以及输出的多样性）。有趣的是，通过对每个level使用不同的\Ψ，在仿射变换块之前，模型可以控制每组特征的平均距离，如下面的视频所示。


### Fine-Tuning
> 在ProGAN上对StyleGAN的进一步改进是更新了几个网络超参数，例如训练周期和损失函数，并将最近邻居的up/downscaling替换为双线性采样。虽然这一步骤对于模型性能很重要，但是它的创新性较差，因此这里不再详细描述（本文附录C）。

![An overview of StyleGAN](https://github.com/GuoYL36/other/tree/master/paper/img/NVIDIA-Style-based-GANs-Chart.png)


### Feature disentanglement
> 为了使关于特征分离的讨论更具量化性，本文提出了两种测量特征解纠缠的新方法：1、感知路径长度 - 在两个随机输入之间进行插值时测量连续图像（VGG16嵌入）之间的差异。剧烈的变化意味着多个特征一起发生了变化，并且表示他们是纠缠的；2、线性可分性 - 将输入进行二分类，例如男性和女性。分类越好，特征就越可分离。通过比较输入向量z和中间向量w的metrics，作者表明w中的特征明显更加可分。这些metrics还显示了与1或2层相比，在Mapping Network中选择8个layers的好处。


## 推荐系统

### deep neural networks for youtube recommendations
+ 特征
    + - word2vec：用来学习video ID的embedding
    + - 用户观看历史：由video ID组成，embedding取平均
    + - 搜索历史：对query进行unigrams tokenize和bigrams tokenize，然后对token取平均
    + - 人口结构特征：
    + - 地理区域：embedded
    + - 设备: embedded
    + - 用户性别、登录状态、年龄(将值归一化)：直接输入网络
+ 3.3节
    + （1）训练好的模型如何处理implicit bias现象？业务指标
    + （2）对于不稳定的流行视频，训练时采用“age”，预测时将“age”设置为0或者负数？这个是常规操作，类似于特征处理时的空值处理方式
    + 推荐涉及解决“代理问题”和将结果转换成特定内容？“代理问题”：用户不是通过youtube官网直接观看的，而是在第三方观看，所以训练集里都用了这些数据

+ 为什么不直接用逻辑回归，而用加权逻辑回归？为什么预测时使用的加权逻辑回归，而serving时是用e^(wx+b)预测时长？
    + 
+ serving函数为什么是e^(wx+b)？
    + odds = p/(1-p)，这里p=sigmoid，log(p/(1-p))=wx+b，所以如果training使用p/(1-p)，serving就可以使用e^(wx+b)
+ 为什么serving函数是预测用户的观看时长？
    + 首先为什么要预测用户观看时长，是因为认为video的观看时长才最能代表用户是否喜好video
    

+ 为什么准备数据的时候，是从每个用户随机抽取定量的数据？
    + 为了消除活跃用户的影响：因为如果直接取用户数据的话，活跃用户数据量大，导致损失倾向于活跃用户；

+ 为什么不直接用mse来直接优化时长，而是用odds？
    + 因为实验发现，用odds能加快收敛。
    + 如果用mse的话，优化时长会出现观看时长为负数的情况，这种情况按理说可以限制负数直接为0，但这样也会有问题？
        + “如果预测为负值，可以直接置为0”，这么做会让所有预测为负值的样本得分都一样，无法排序，也无法在用于后面的业务流程，信息损失比较大。
+ 训练例子：来自于所有youtube的观看视频，而不仅仅是在推荐上的视频


### deep interest network(DIN)
+ 创新点：
    +（1）目前embedding学习的长度都是固定的，但是为了更好的学习不同人的兴趣，这里使用局部激活单元(local activation unit)来更好的学习表征
    +（2）优化训练，减少训练参数：i) mini-batch aware regularizer，节约正则开销，避免过拟合；ii) data adaptive activation function
+ 特征表示
    + multi-group categorial form
        + 比如：[weekday=Friday, gender=Female, visited_cate_ids={Bag,Book}, ad_cate_id=Book]
        + 两种形式：one-hot、multi-hot
    + embeddings
        + one-hot：对应一个向量；
        + multi-hot：对应多个向量；(利用pooling层将embedding vector list转换成固定长度向量)
    + multi-hot的embedding最后变成固定长度向量限制了用户不同兴趣的表达能力
        + **引入local activation unit**：对每个用户行为embedding向量加权重（计算每个向量与广告向量之间的相似度）
            + 类似attention，但是这里权重之和会大于1
                + 为什么attention里最后的权重都归一化了，而这里没有归一化？
                + 文章中解释说不归一化的目的是一定程度上近似保留用户兴趣意图，我这里的理解：文章中说将权重总和用来近似该广告激发用户兴趣的程度，可能是因为不同广告激发程度不一样，如果归一化的话，那所有广告对用户的激发程度就是一样了。
                + 不归一化又引出一个问题：因为归一化的目的是使数据相对集中，如果不归一化的话，数据的差异性估计会很大，这个怎么处理？
                + 猜测后面的fn有归一化操作
+ 模型优化
    + mini-batch aware regularizer，节约正则开销，避免过拟合；
        + 由于模型中还会输入一些细粒度特征，比如goods_id，如果没有正则，会导致模型在训练完第一个epoch时性能急剧下降。
        + 由于稀疏输入和上亿参数，不适合直接使用l1和l2正则？
            + 用l2正则举例：由于l2正则是直接应用于所有参数，而模型中只需要对一部分非零稀疏特征的参数需要更新，这会造成计算开销很大。
            + 本文只在稀疏特征的参数上计算l2正则
    + data adaptive activation function
        + PReLU激活函数在值为0处是个硬调整点(hard rectified point)，不适合当每层输入都符从不同分布的情况
        + dice激活函数：根据输入数据的分布自适应调整rectified point
            + f = p(x)*x+(1-p(x))*a*x   p(x)=1/(1+e^(-(x-E[x])/sqrt(Var[x]+epsilon))), epsilon=1e-8，训练阶段，E[x]和Var[x]是mini-batch计算得到，测试阶段，E[x]和Var[x]是滑动平均。
            + 相比PReLU，Dice激活函数更平滑

+ tricks:针对大数据
    + Dropout：每个样本随机丢弃50%的feature ids
    + Filter：通过共现频率过滤visited goods_id，仅留下最高频的；本文中设置前2亿good_ids留下
    + Regularization in DiFacto：与frequent features相关的参数很少over-regularized
    + MBA：Mini-Batch Aware正则，正则参数值设置为0.01
+ 问题

### deep interest extract network
+ 创新点：
    + 对“兴趣进化现象(interest evolving phenomenon)”进行建模
    + DIN是直接把行为当作兴趣，而本文设计了一个兴趣捕获层。由于GRU的hidden state是很少专注兴趣表征，所以这里增加auxiliary loss，使用连续行为去监督每一步hidden state学习；
    + 设计interest evolving layer，使用AUGRU(attention update gate)强化与目标item的兴趣，克服兴趣漂移(interest drifting)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    







