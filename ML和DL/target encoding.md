# Target Encoding 

------
> <font color=red>将类型变量（categorical variables）转换成连续的数值表示</font>
[链接](https://maxhalford.github.io/blog/target-encoding-done-the-right-way/)

## 可以选择以下方法：
- Label Encoding：为每一个类型变量选择一个随机数，如1,2,3，。。。
- One-hot encoding：将Label Encoding表示的类型变量值转换成二进制形式，如[1,0,0],[0,1,0]
- Vector representation：例如word2vec，寻找低维空间拟合数据
- Optimal binning：利用树学习器，如LightGBM、CatBoost
- Target encoding：根据类型对目标值求平均来表示

> 上述任何一种方法都存在优点和缺点，取决于数据和要求。如果变量有很多类型且使用one-hot encoding，会造成内存问题（稀疏问题），可以尝试使用Optimal binning。Label encoding最好别使用，除非碰到类型变量是ordinal的情况，即“cold”为0, “mild”为1, and “hot”为2。word2vec和其它方法也很好，不过需要微调。

### Target encoding
> 类型变量x,标签y。对于类型变量中的每一个不同的x，根据标签y值计算平均值，利用这个平均值来替代x。
```python
import pandas as pd

df = pd.DataFrame({
	'x_0':['a']*5+['b']*5,
	'x_1':['c']*9+['b']*1,
	'y':[1,1,1,1,0,1,0,0,0,0]
})
```

|    x_0    |    x_1    |    y    |
| --------- | --------- | ------- |
|    a      |     c     |    1    |
|    a      |     c     |    1    |
|    a      |     c     |    1    |
|    a      |     c     |    1    |
|    a      |     c     |    0    |
|    b      |     c     |    1    |
|    b      |     c     |    0    |
|    b      |     c     |    0    |
|    b      |     c     |    0    |
|    b      |     d     |    0    |

```python
means = df.groupby('x_0')['y'].mean()
# means
	{
	'a': 0.8,    # 4/5
	'b': 0.2     # 1/5
	}
```
```python
df['x_0'] = df['x_0'].map(df.groupby('x_0')['y'].mean())
df['x_1'] = df['x_1'].map(df.groupby('x_1')['y'].mean())
```

|    x_0    |    x_1    |    y    |
| --------- | --------- | ------- |
|    0.8      |     0.555     |    1    |
|    0.8      |     0.555     |    1    |
|    0.8      |     0.555     |    1    |
|    0.8      |     0.555     |    1    |
|    0.8      |     0.555     |    0    |
|    0.2      |     0.555     |    1    |
|    0.2      |     0.555     |    0    |
|    0.2      |     0.555     |    0    |
|    0.2      |     0.555     |    0    |
|    0.2      |     0     |    0    |

#### 存在问题：over-fitting。因为无法保证被替代的类型变量值适用于测试集中。例如类型变量x_1中d值被平均值0替代，因为它只出现1次且相应的y值为0，所以出现了过拟合，因为没有足够的数据保证0可以用来替代d值。
#### 解决over-fitting的方法：
- [x]交叉验证并对结果求平均；
- [x]additive smoothing;

```python
def calc_smooth_mean(df, by, on, m):
	# Compute the global mean
	mean = df[on].mean()
	
	# Compute the number of values and the mean of each group
	agg = df.groupby(by)[on].agg(['count', 'mean'])
	counts = agg['count']
	means = agg['mean']
	
	# Compute the "smoothed" means
	smooth = (counts * means + m * mean) / (counts + m)
	
	# Replace each value by the according smoothed mean
	return df[by].map(smooth)

df['x_0'] = calc_smooth_mean(df, by='x_0', on='y', m=10)
df['x_1'] = calc_smooth_mean(df, by='x_1', on='y', m=10)

```

|    x_0    |    x_1    |    y    |
| --------- | --------- | ------- |
|    0.6      |     0.526316     |    1    |
|    0.6      |     0.526316     |    1    |
|    0.6      |     0.526316     |    1    |
|    0.6      |     0.526316     |    1    |
|    0.6      |     0.526316     |    0    |
|    0.4      |     0.526316     |    1    |
|    0.4      |     0.526316     |    0    |
|    0.4      |     0.526316     |    0    |
|    0.4      |     0.526316     |    0    |
|    0.4      |     0.454545     |    0    |

