# tensorflow中的函数使用

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

















































