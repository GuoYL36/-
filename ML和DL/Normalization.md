
## BN、LN、IN、GN、WS

> **BN和GN关注于激活值的归一化，而WS关注于权重的平滑效果**
> **加速模型训练和收敛，提升效果**

### BN

#### 方法有效的解释：
+ 能缓解内部协变量转移，即 Internal Covariate Shift，ICS；
+ 和ICS减少无关，而是使得优化问题的曲线更平滑；

#### 用法：
+ 在Batch维度上进行Normalization，归一化维度为[N, H, W]，对Batch中对应的Channel归一化；
+ 对Batch size有依赖，仅在Batch Size足够大时才有明显的效果，因此不能用在微Batch Size的训练中；

```python
# BN代码实现
# 输入维度：[N, C, H, W]，其中x为[N, C×H×W]，N是Batch size，H/W是feature的高/宽，C是feature的Channel；
mu = np.mean(x, axis=0)
sigma = np.var(x, axis=0)
x_hat = (x - mu)/np.sqrt(sigma + eps)
out = gamma * x_hat + beta

```

### LN
+ LN避开了Batch维度，归一化维度为[C, H, W]
```python
# LN的前向传播的代码实现，其中x为[N, C×H×W]
x = x.T
mu = np.mean(x, axis=0)
sigma = np.var(x, axis=0)
x_hat = x_hat.T
out = gamma * x_hat + beta
inv_sigma = 1. / np.sqrt(sigma + eps)
cache = (x_hat, gamma, mu, inv_sigma)

```

### IN
+ 归一化维度为[H, W]

### LN、IN在训练RNN/LSTM等递归模型，或者GAN等生成模型方面特别成功。

### GN

> GN介于LN、IN之间，首先将Channel分为许多group，对每一group做Normalization，以及先将feature的维度由[N, C, H, W] reshape为[N*G, C//G, H, W]，归一化的维度为[C//G, H, W]；<br>
> GN的极端情况为LN和IN，即LN为G=1，IN为G=C

#### 方法目的：
+ 解决BN中微Batch Size训练的问题，但是对于Batch Size比较大时，效果不如BN；

### WS
+ WS方法可以减少损失和梯度的[Lipsschitz常数](https://blog.csdn.net/victoriaw/article/details/58006629 "Lipsschitz常数")。因此，它能平滑损失曲线并提升训练效果。
+ 对于能够使用大批次的任务，使用Batch Size为1的GN+WS，其效果能够比肩甚至超过大批次下BN的效果。
+ 对于微Batch Size，GN+WS可以大幅提升效果。
+ WS方法可以帮助更轻松的训练深层次网络，而不用担心内存和Batch Size问题。


## Xavier
> **原理：**根据输入和输出神经元的数量自动决定初始化的范围，假设参数所在层的输入维度为 n，输出维度为 m，则参数将从 [-sqrt(6/(m+n)), sqrt(6/(m+n))]均匀分布中采样。

+ **作用：**
	+ 帮助减少梯度弥散问题；

## MSRA Filler
> **原理：**当使用Xavier初始化时，如果激活函数为ReLU，会使得输出中一半数据为0，这时这一半数据的梯度也为0，而tanh和sigmoid等激活函数的输出初始化时梯度接近于1. 因此，使用ReLU的网络参数的方差可能会带来波动。为避免这种情况，使用均值为0，方差为sqrt(4/(n_in+n_out))的高斯分布，其中 n_in 为输入维度，n_out 为输出维度。






























