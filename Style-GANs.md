
## Style-based GANs-Generating and Tuning Realistic Artifical Faces
> [Style-based GANs](https://www.lyrn.ai/2018/12/26/a-style-based-generator-architecture-for-generative-adversarial-networks/ "Style-based GANs")


[ProGAN overview](https://www.lyrn.ai/wp-content/uploads/2018/12/ProGAN-chart-1.png "ProGAN overview")

> ProGAN生成高质量图片，但是和很多模型一样，控制生成图片的特定特征是有限的。换句话说，特征是纠缠的，如果对输入稍稍调整，就会影响多个特征。打个比方，现实中只要改变一个基因就会影响多个特征。
[ProGAN progressive training from low to high res layers](https://www.lyrn.ai/wp-content/uploads/2018/12/ProGAN.gif "ProGAN progressive training from low to high res layers")


### How StyleGAN works
> StyleGAN文章提供了ProGAN的图片生成器的更新版本，主要关注生成器网络。文中作者发现ProGAN中progressive layer的潜在优势是如果能得到合理利用，那么就能够控制图片的不同视觉特征。层数越少(分辨率越低)，它影响越Coarser(粗糙的)特征。这篇文章将特征分成三类：1、Coarse(粗糙)——分辨率达到8×8——影响姿态、general头发类型、脸部形状等；2、Middle——分辨率为16×16到32×32——影响好的脸部特征、头发类型、睁眼或闭眼等；3、Fine——分辨率为64×64到1024×1024——影响颜色方案(眼睛、头发和皮肤)和微特征。



### Mapping Network
> Mapping Network的目的是将输入向量编码成中间向量，利用这个中间向量的不同元素控制着不同的视觉特征。这是一个non-trivial过程，因为直接控制输入向量的视觉特征是受限的，它必须服从训练数据的概率密度。<br>例如，如果数据集中黑头发人物图片更普遍，那么更多的输入值将被映射成黑头发的特征。结果，模型不能够将部分输入(向量元素)映射成特征，这个现象称作为特征纠缠。然而，通过使用另一神经网络，模型能够生成一个不需要服从训练数据分布的向量，这个向量能够减少特征之间的相关性。<br>Mapping Network由8个全连接层组成，其输出w的维度和输入维度(512×1)一致。

[The generator with the Mapping Network(in addition to the ProGAN synthesis network)](https://www.lyrn.ai/wp-content/uploads/2018/12/StyleGAN-generator-Mapping-network.png "The generator with the Mapping Network(in addition to the ProGAN synthesis network)")

### Style Modules(AdaIN)
> [AdaIN](https://arxiv.org/abs/1703.06868 "AdaIN")(Adaptive Instance Normalization)模块将编码信息w(来自Mapping-network)转变为已生成图片。这个模块被增加到Synthesis网络的每一个分辨率层，并在那一层定义特征的视觉表示：1、卷积层输出的每一个channel首先进行归一化，确保步骤3中的scaling和shifting有期望的效应；2、使用另一个全连接层(标记为 A )将中间向量w变换为每一个channel的scale和bias；3、scale和bias向量调整卷积输出的每个channel，可以认为是卷积的每个滤波器的权重。这个调整可以将来自w的信息翻译成视觉表示。

[The generator's Adaptive Instance Normalization (AdaIN)](https://www.lyrn.ai/wp-content/uploads/2018/12/StyleGAN-generator-AdaIN.png "The generator's Adaptive Instance Normalization (AdaIN)")

### Removing traditional input
> 很多模型，包括ProGAN都使用随机输入创建生成器的初始图片(相当于4×4的水平)。StyleGAN团队发现图片特征由w和AdaIN控制，因此初始输入会被省略和由常数值替代。虽然文章中没有解释为什么能提升性能，但是一个假设是它能够减少特征纠缠——对于网络来说，它更容易仅仅使用不依赖于纠缠的输入向量的w去学习。

[The Synthesis Network input is replaced with a constant input](https://www.lyrn.ai/wp-content/uploads/2018/12/StyleGAN-generator-Input.png "The Synthesis Network input is replaced with a constant input")


### Stochastic variation
> 人类脸上有很多方面是微小的并且能被随机地观察到，比如：雀斑、头发的准确放置、皱纹，这些特征使得图片更加真实和增加输出的多样性。将这些小特征插入GAN图片的常用方法是将随机噪声加入输入向量中。然而，在许多例子中，由于特征纠缠现象会导致图片的其他特征受影响，所以控制噪声效应是很复杂的。StyleGAN中的噪声以相同的方式被加到AdaIN机制——一种Scaled噪声被加入AdaIN模块之前的每一个channel中，并改变了一些分辨率层中特征的视觉表示。

[Adding scaled noise to each resolution level of the synthesis network](https://www.lyrn.ai/wp-content/uploads/2018/12/StyleGAN-generator-Noise.png "Adding scaled noise to each resolution level of the synthesis network")

### Style mixing
> StyleGAN生成器使用了Synthesis Network中每一层的中间向量，这可能导致网络学习到的levels是相关的。为了减少相关性，模型随机选择两个输入向量并为它们生成中间向量w。然后它对第一个中一些levels进行训练并且（在一个随机点上）切换到另一个其它以训练其余的levels。随机切换确保网络将不会学习并依赖于levels之间的相关性。虽然在所有数据集上，它没有改善模型性能，但这个概念有一个非常有趣的副作用 - 它能够以连贯的方式组合多个图像（如下面的视频所示）。该模型生成两个图像A和B，然后通过从A中获取低level特征并从B中获取其余特征来组合它们。

### Truncation trick in W
> 生成模型面临的挑战之一是处理训练数据中表现不佳的领域。生成器无法学习它们并创建类似于它们的图像（而是创建看起来很糟糕的图像）。为了避免生成不良图像，StyleGAN截断中间向量w，强制它接近“平均”中间向量。训练模型后，通过选择许多随机输入产生“平均”w_avg; 用Mapping Network生成它们的中间向量; 并计算这些向量的平均值。在生成新图像时，不是直接使用Mapping Network输出，而是将w转换为w_new = w_avg +\Ψ（w-w_avg），其中\Ψ的值定义图像距“平均”图像的距离（以及输出的多样性）。有趣的是，通过对每个level使用不同的\Ψ，在仿射变换块之前，模型可以控制每组特征的平均距离，如下面的视频所示。


### Fine-Tuning
> 在ProGAN上对StyleGAN的进一步改进是更新了几个网络超参数，例如训练周期和损失函数，并将最近邻居的up/downscaling替换为双线性采样。虽然这一步骤对于模型性能很重要，但是它的创新性较差，因此这里不再详细描述（本文附录C）。

[An overview of StyleGAN](https://www.lyrn.ai/wp-content/uploads/2018/12/NVIDIA-Style-based-GANs-Chart.png "An overview of StyleGAN")


### Feature disentanglement
> 为了使关于特征分离的讨论更具量化性，本文提出了两种测量特征解纠缠的新方法：1、感知路径长度 - 在两个随机输入之间进行插值时测量连续图像（VGG16嵌入）之间的差异。剧烈的变化意味着多个特征一起发生了变化，并且表示他们是纠缠的；2、线性可分性 - 将输入进行二分类，例如男性和女性。分类越好，特征就越可分离。通过比较输入向量z和中间向量w的metrics，作者表明w中的特征明显更加可分。这些metrics还显示了与1或2层相比，在Mapping Network中选择8个layers的好处。

