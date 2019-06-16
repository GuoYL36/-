# tensorflow中的函数使用

- [content]
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












































