# python基础

- [目录](#)
	- [list和array的底层实现以及区别](#list和array的底层实现以及区别)
	- [函数](#函数)
		- [函数中的各种参数](#函数中的各种参数)
	- [DataFrame](#DataFrame)
	- [编码](#编码)
		- [ASCII](#ASCII)
		- [ANSI](#ANSI)
		- [MBCS](#MBCS)
		- [GB2312](#GB2312)
		- [GBK](#GBK)
		- [Unicode](#Unicode)
		- [UTF-8](#UTF-8)
		- [python中文乱码问题](#python中文乱码问题)	
	- [类继承](#类继承)
		- [super的作用](#super的作用)
		- [访问可见性](#访问可见性)
		- [装饰器](#装饰器)
		- [slots魔法](#slots魔法)
		- [静态方法和类方法](#静态方法和类方法)
		- [抽象类](#抽象类)
		- [内建元类type](#内建元类type)
	- [进程、线程、协程](#进程和线程和协程)
		- [进程](#进程)
		- [线程](#线程)
		- [协程](#协程)
		- [对比](#对比)
		- [GIL锁](#GIL锁)
	- [闭包](#闭包)


[python内存可视化](http://www.pythontutor.com/visualize.html#mode=display)


## list和array的底层实现以及区别

+ python中list的底层实现
	+ list属于连续线性表，有一块连续地内存存储list中元素的地址；
	+ list最大长度：2^(29)，即32位；或2^(60)，即64位；
	+ list对象中包含指向list对象的指针数组和申请内存槽的个数。初始化一个空list时，会在内存中分配好内存槽空间数。一般槽的大小要大于list的大小(len)。当list为空时，分配的内存槽大小为0；当进行append操作添加一个元素时，分配的内存槽大小为4，然后直到list长度等于内存槽大小时，继续添加一个元素，此时又会重新分配一个大小为8的内存槽，依次下去（list的增长模式为：0、4、8、16、25、35、…）。当使用pop操作时，如果取出list中元素后的大小小于分配的内存槽大小的一半，那么将会缩减list的大小(缩减到当前list大小的2倍，例如：分配的内存槽大小为8，当list中剩下3个元素时，内存槽大小缩减到6)。

+ list和array的区别
	+ array中数据类型时一样的，而list中可以有各种类型数据；
	+ array一般计算起来比list快，因为list涉及到类型的转换问题；
	
## 函数

### 函数中的各种参数
> 定义和调用时不同类型参数的顺序：<br>		1. 位置参数<br>		2. 默认参数<br>		3. 可变参数<br>		4. 关键字参数.

+ 位置参数：根据函数定义的参数位置传递参数；
+ 可变参数：用一个*表示，根据传进参数位置合并为一个元组；
+ 关键字参数：用**表示，通过“键-值”形式加以指定，不需要考虑参数顺序；
+ 默认参数：函数定义时，提供了默认参数值.


## DataFrame

### ix、loc、iloc、at、iat
> DataFrame对象的索引方式

+ **ix：** 获取某个位置的值或者切片操作
	+ 既可以使用index索引值和columns索引值进行索引，也可以使用下标直接索引；

+ **loc、iloc：** 获取某个位置的值或者切片操作
	+ loc 利用DataFrame的index索引值和columns索引值进行索引；
	+ iloc 利用下标进行索引；

+ **at、iat：** 获取某个位置的值
	+ at 利用DataFrame的index索引值和columns索引值进行索引；
	+ iat 利用下标进行索引；

	
## 编码

[感谢joyfixing—彻底搞懂python中文乱码问题](#https://blog.csdn.net/joyfixing/article/details/79971667)

### ASCII
>   美国信息互换标准代码（American Standard Code for Information Interchange），使用8个bit位（一个字节）进行编码，用不同字节来存储英语的文字。**不支持中文编码**，这也是windows里不支持中文显示的原因。

### ANSI
>   美国国家标准协会(American National Standard Institite)，也就是说，每个国家（非拉丁语系国家）自己制定自己的文字的编码规则，并得到了ANSI认可，符合ANSI的标准，全世界在表示对应国家文字的时候都通用这种编码就叫ANSI编码。换句话说，中国的ANSI编码和在日本的ANSI的意思是不一样的，因为都代表自己国家的文字编码标准。比如中国的ANSI对应就是GB2312标准，日本就是JIT标准，香港，台湾对应的是BIG5标准等等。当然这个问题也比较复杂，微软从95开始，用就是自己搞的一个标准GBK。GB2312里面只有6763个汉字，682个符号，所以确实有时候不是很够用。GBK一直能和GB2312相互混淆并且相安无事的一个重要原因是GBK全面兼容GB2312，所以没有出现任何冲突，你用GB2312编码的文件通过GBK去解释一定能获得相同的显示效果，换句话说：GBK对GB2312就是，你有的，我也有，你没得的，我还有！好了，ANSI的标准是什么呢，首先是ASCII的代码你不能用！也就是说ASCII码在任何ANSI中应该都是相同的。其他的，你们自己扩展。所以呢，中国人就把ASCII码变成8位，0x7f之前我不动你的，我从0xa0开始编，0xa0到0xff才95个码位，对于中国字那简直是杯水车薪，因此，就用两个字节吧，因此编码范围就从0xA1A1 - 0xFEFE，这个范围可以表示23901个汉字。基本够用了吧，GB2312才7000多个呢！GBK更猛，编码范围是从0x8140 - 0xFEFE,可以表示3万多个汉字。可以看出，这两种方案，都能保证汉字头一个字节在0x7f以上，从而和ASCII不会发生冲突。能够实现英文和汉字同时显示。
> BIG5，香港和台湾用的比较多，繁体，范围： 0xA140 - 0xF9FE, 0xA1A1 - 0xF9FE，每个字由两个字节组成，其第一字节编码范围为0xA1~0xF9，第二字节编码范围为0x40-0x7E与0xA1-0xFE，总计收入13868个字 (包括5401个常用字、7652 个次常用字、7个扩充字、以及808个各式符号)。
> 那么到底ANSI是多少位呢？这个不一定！比如在GB2312和GBK，BIG5中，是两位！但是其他标准或者其他语言如果不够用，就完全可能不止两位！例如：GB18030:GB18030-2000(GBK2K)在GBK的基础上进一步扩展了汉字，增加了藏、蒙等少数民族的字形。GBK2K从根本上解决了字位不够，字形不足的问题。它有几个特点：它并没有确定所有的字形，只是规定了编码范围，留待以后扩充。编码是变长的，其二字节部分与GBK兼容；四字节部分是扩充的字形、字位，其编码范围是首字节0x81-0xfe、二字节0x30-0x39、三字节0x81-0xfe、四字节0x30-0x39。它的推广是分阶段的，首先要求实现的是能够完全映射到Unicode3.0标准的所有字形。它是国家标准，是强制性的。搞懂了ANSI的含义，我们发现ANSI有个致命的缺陷，就是每个标准是各自为阵的，不保证能兼容。换句话说，要同时显示中文和日本文或者阿拉伯文，就完全可能会出现一个编码两个字符集里面都有对应，不知道该显示哪一个的问题，也就是编码重叠的问题。显然这样的方案不好，所以Unicode才会出现！

### MBCS
> （Multi-Byte Chactacter System（Set)） 多字节字符系统或者字符集，基于ANSI编码的原理上，对一个字符的表示实际上无法确定他需要占用几个字节的，只能从编码本身来区分和解释。因此计算机在存储的时候，就是采用多字节存储的形式。也就是你需要几个字节我给你放几个字节，比如A我给你放一个字节，比如"中“，我就给你放两个字节，这样的字符表示形式就是MBCS。
> 在基于GBK的windows中，不会超过2个字节，所以windows这种表示形式有叫做DBCS（Double-Byte Chactacter System），其实算是MBCS的一个特例。
C语言默认存放字符串就是用的MBCS格式。从原理上来说，这样是非常经济的一种方式。


### GB2312
>   由于除美国之外的其他国家有自己的文字，因此他们采用127号之后的空位来表示新的字母、符号到等，一直把序号编到了最后一个状态255.从188到255这一页的字符集被称“扩展字符集”。
>   GB2312是对ASCII的中文扩展，把127号之后的奇异符号取消了，规定：一个小于127的字符的意义与原来相同，但两个大于127的字符连在一起时，就表示一个汉字，高字节（第一个字节）从0xA1到0xF7，低字节（第二个字节）从0xA1到0xFE，这样可以组合大约7000多个简体汉字。在这些编码里，我们还把数学符号、罗马希腊的字母、日文的假名们都编进去了，连在 ASCII 里本来就有的数字、标点、字母都统统重新编了两个字节长的编码，这就是常说的”全角”字符，而原来在127号以下的那些就叫”半角”字符了。

### GBK
> 但是中国的汉字太多了，我们很快就就发现有许多人的人名没有办法在这里打出来，特别是某些很会麻烦别人的国家领导人。于是我们不得不继续把 GB2312没有用到的码位找出来老实不客气地用上。后来还是不够用，于是干脆不再要求低字节一定是127号之后的内码，只要第一个字节是大于127就固定表示这是一个汉字的开始，不管后面跟的是不是扩展字符集里的内容。结果扩展之后的编码方案被称为GBK标准，GBK包括了GB2312的所有内容，同时又增加了近20000个新的汉字（包括繁体字）和符号。后来少数民族也要用电脑了，于是我们再扩展，又加了几千个新的少数民族的字，GBK 扩成了 GB18030。
>  中国的程序员们看到这一系列汉字编码的标准是好的，于是通称他们叫做 “DBCS“（Double Byte Charecter Set 双字节字符集）。在DBCS系列标准里，最大的特点是**两字节长的汉字字符**和一字节长的英文字符并存于同一套编码方案里，因此他们写的程序为了支持中处理，必须要注意字串里的每一个字节的值，如果这个值是大于127的，那么就认为一个双字节字符集里的字符出现了。

### Unicode
> ISO（国际标准化组织）的国际组织废了所有的地区性编码方案，重新搞一个包括了地球上所有文化、所有字母和符号的编码！他们打算叫它”Universal Multiple-Octet Coded Character Set”，简称 UCS, 俗称“unicode“。unicode开始制订时，计算机的存储器容量极大地发展了，空间再也不成为问题了。于是 ISO就直接规定必须用两个字节，也就是16位来统一表示所有的字符，对于ASCII里的那些“半角”字符，unicode包持其原编码不变，只是将其长度由原来的8位扩展为16位，而其他文化和语言的字符则全部重新统一编码。
> 由于”半角”英文符号只需要用到低8位，所以其高8位永远是0，因此这种大气的方案在保存英文文本时会多浪费一倍的空间。这时候，从旧社会里走过来的程序员开始发现一个奇怪的现象：他们的 strlen 函数靠不住了，一个汉字不再是相当于两个字符了，而是一个！是的，从 unicode开始，无论是半角的英文字母，还是全角的汉字，它们都是统一的”一个字符“！同时，也都是统一的”两个字节“，请注意”字符”和”字节”两个术语的不同，“字节”是一个8位的物理存贮单元，而“字符”则是一个文化相关的符号。在 unicode中，一个字符就是两个字节。一个汉字算两个英文字符的时代已经快过去了。
> unicode 同样也不完美，这里就有两个的问题，一个是，如何才能区别 unicode和ASCII？计算机怎么知道三个字节表示一个符号，而不是分别表示三个符号呢？第二个问题是，我们已经知道，英文字母只用一个字节表示就够了，如果unicode统一规定，每个符号用三个或四个字节表示，那么每个英文字母前都必然有二到三个字节是0，这对于存储空间来说是极大的浪费，文本文件的大小会因此大出二三倍，这是难以接受的。

### UTF-8
> unicode 在很长一段时间内无法推广，直到互联网的出现，为解决 unicode 如何在网络上传输的问题，于是面向传输的众多 UTF（UCS Transfer Format）标准出现了，顾名思义，UTF-8就是每次8个位传输数据，而 UTF-16 就是每次16个位。UTF-8就是在互联网上使用最广的一种 unicode 的实现方式，这是为传输而设计的编码，并使编码无国界，这样就可以显示全世界上所有文化的字符了。UTF-8最大的一个特点，就是它是一种变长的编码方式。它可以使用1~4个字节表示一个符号，根据不同的符号而变化字节长度，当字符在 ASCII 码的范围时，就用一个字节表示，保留了 ASCII 字符一个字节的编码做为它的一部分，注意的是 **unicode一个中文字符占2个字节**，**而UTF-8一个中文字符占3个字节**）。从 unicode 到 uft-8并不是直接的对应，而是要过一些算法和规则来转换。

### python中文乱码问题
+ 查看编码格式：sys.getdefaultencoding()
+ 解决中文显示乱码问题：**确保python文件中的中文编码格式与系统的中文编码格式一致**
	+ 第一种方法：打印的时候进行转码操作
	```python
	# encoding: utf-8
	
	import sys
	type = sys.getfilesystemencoding()
	print myname.decode('UTF-8').encode(type)
	
	```
	+ 文件存储为utf-8格式，编码声明utf-8，即：文件头加上# encoding:utf-8
	+ 出现汉字的地方前面加上u
	+ 不同编码之间不能直接转换，要经过 unicode 中间跳转
	+ cmd下不支持utf-8编码
	+ raw_input提示字符串只能为gbk编码
+ 编解码转换：utf-8和gbk不能直接编转码，需经unicode中间跳转
	+ 编码：从unicode开始是编码过程，使用encode，例：unicode —> gbk，u.encode('gbk')
	+ 解码：其他编码转换为unicode，使用decode，例：utf-8 —> unicode，"中文".decode('utf-8')
+ raw_input
	+ raw_input是获取用户输入值的，获取到的用户输入值和当前运行环境编码有关，比如 cmd 下默认编码是gbk，那么输入的汉字就是以gbk编码，而不管 python文件编码格式和编码声明。

----

## 类继承

### super的作用
> 当父类多次被子类调用时，只执行了一次，优化了执行逻辑，解决子类调用父类方法的一些问题。

+ 例子
	+ 子类中调用父类的最简单方法为
		```python
		class FooParent:                # 父类
			def bar(self, message):
				print(message)
	
		class FooChild(FooParent):          # 子类
			def bar(self, message):
				FooParent.bar(self, message)
	
		>>> FooChild().bar("Hello, Python.")    # 结果：Hello, Python
	
		```
	+ 上述方法存在问题：
		+ 如果父类名称修改，则所有继承的子类都要修改。因此，python引入super()机制。
			```python
			class FooParent:                # 父类
				def bar(self, message):
					print(message)
	
			class FooChild(FooParent):          # 子类
				def bar(self, message):
					super(FooChild, self).bar(message)
	
			>>> FooChild().bar("Hello, Python.")    # 结果：Hello, Python
	
			```
		+ 并且在多继承中，由于内部处理机制不同，简单方法的父类会被多次执行。
			```python
			# 简单方法的父类会被多次执行：
			class A:
				def __init__(self):
					print("Enter A")
					print("Leave A")
					
			class B(A):
				def __init__(self):
					print("Enter B")
					A.__init__(self)
					print("Leave B")
					
			class C(A):
				def __init__(self):
					print("Enter C")
					A.__init__(self)
					print("Leave C")
					
			class D(A):
				def __init__(self):
					print("Enter D")
					A.__init__(self)
					print("Leave D")
					
			class E(B,C,D):
				def __init__(self):
					print("Enter E")
					B.__init__(self)
					C.__init__(self)
					D.__init__(self)
					print("Leave E")
					
			>>> E()
			结果:
				Enter E
				Enter B
				Enter A
				Leave A
				Leave B
				Enter C
				Enter A
				Leave A
				Leave C
				Enter D
				Enter A
				Leave A
				Leave D
				Leave E
			
			# super机制里可以保证公共父类仅被执行一次，至于执行的顺序按照MRO方法解析顺序执行。
			class A:
				def __init__(self):
					print("Enter A")
					print("Leave A")
					
			class B(A):
				def __init__(self):
					print("Enter B")
					super(B, self).__init__()
					print("Leave B")
					
			class C(A):
				def __init__(self):
					print("Enter C")
					super(C, self).__init__()
					print("Leave C")
					
			class D(A):
				def __init__(self):
					print("Enter D")
					super(D, self).__init__()
					print("Leave D")
					
			class E(B,C,D):
				def __init__(self):
					print("Enter E")
					super(E, self).__init__()
					print("Leave E")
					
			>>> E()			
			结果：
				Enter E
				Enter B
				Enter C
				Enter D
				Enter A
				Leave A
				Leave D
				Leave C
				Leave B
				Leave E
			
			```

### 访问可见性
+ **公开的**
	+ 系统定义名字：前后均有一个双下划线


+ **保护的**
	+ 命名时可以用一个单划线作为开头：_xxx
	
+ **私有的**
	+ 命名时可以用两个下划线作为开头：__xxx
	+ 访问私有成员变量的正确方式：
		+ 私有变量：实例对象名._类名__变量名
		+ 私有方法：实例对象名._类名__方法名()

	```python
	class Test:
		def __init__(self, foo):
			self.__foo = foo    # 私有属性
			self.name = foo     # 公开属性
	
		def __bar(self):        # 私有方法
			print(self.__foo)     
			print(self.name)
			print('__bar')
	
		def bar(self):          # 公有方法
			print(self.__foo)
			print('__bar')
		
	# def main():
		# test = Test("hello")
		# test.bar()
		#test.__bar()    # 报错AttributeError: 'Test' object has no attribute '__bar'
		# print(test.__foo)  # 报错AttributeError: 'Test' object has no attribute '__foo'
	
	def main():
		test = Test("hello")
		test._Test__bar()     # 访问私有方法的方式
		print(test._Test__foo)     # 访问私有变量的方式
	
	if __name__ == "__main__":
		main()

	```	

### 装饰器

> 为了使得**对属性的访问既安全又方便**，考虑使用@property包装器来包装getter和setter方法。
> **对于多个函数，需要为这多个函数增加相同功能，此时，可以利用装饰器实现。**

	```python
	class Person(object):
		def __init__(self, name, age):
			self._name = name
			self._age = age
		
		# 访问器 - getter方法
		@property
		def name(self):
			return self._name
	
		# 访问器 - getter方法	
		@property
		def age(self):
			return self._age
		
		# 修改器 - setter方法
		@age.setter
		def age(self, age):
			self._age = age
		
		def play(self):
			if self._age <= 16:
				print("%s正在玩飞行棋."%self._name)
			else:
				print("%s正在玩斗地主."%self._name)

	def main():
		person = Person("王大锤", 12)
		person.play()
		person.age = 22
		person.play()
		person.name = "白元芳"     # 报错AttributeError：can't set attribute
	
	if __name__ == "__main__":
		main()
	
	```
```python
# 能够支持传递各种参数的装饰器
import time
def runtime(func):
	def get_time(*args,**kwargs):
		print(time.time())
		func(*args, **kwargs)
	return get_time

@runtime
def decorate1(*args,**kwargs):
	print("aaaaaaaaaaaa")

decorate1(1,k=1)

```

### slots魔法

> python是一门动态语言，允许程序运行时给对象绑定新的属性或方法。如果需要限定自定义类型的对象只能绑定某些属性，可以通过在类中定义__slots__变量来进行限定。
> __slots__变量的限定只对当前类的对象生效，对子类并不起任何作用。

	```python
	class Person(object):

		# 限定Person对象只能绑定_name, _age和_gender属性
		__slots__ = ('_name', '_age', '_gender')
		def __init__(self, name, age):
			self._name = name
			self._age = age
		
		# 访问器 - getter方法
		@property
		def name(self):
			return self._name
	
		# 访问器 - getter方法	
		@property
		def age(self):
			return self._age
		
		# 修改器 - setter方法
		@age.setter
		def age(self, age):
			self._age = age
		
		def play(self):
			if self._age <= 16:
				print("%s正在玩飞行棋."%self._name)
			else:
				print("%s正在玩斗地主."%self._name)

	def main():
		person = Person("王大锤", 12)
		person.play()
		person.age = 22
		person.play()
		person._gender = '男'
		print("person gender: %s"%person._gender)
		person._is_gay = True     # 报错AttributeError：'Person' object has no attribute '_is_gay'
	
	if __name__ == "__main__":
		main()
	
	```

### 静态方法和类方法
> 在类中定义的方法并不需要全都是对象方法(发送给对象的消息)，还可以是静态方法和类方法。

+ 静态方法：在对象未创建之前，调用该方法
	```python
	from math import sqrt

	class Triangle(object):
	
		def __init__(self,a,b,c):
			self._a = a
			self._b = b
			self._c = c
		
		# 定义静态方法
		@staticmethod
		def is_valid(a,b,c):
				return a+b>c and b+c>a and a+c>b
		
		def perimeter(self):
			return self._a + self._b + self._c
	
		def area(self):
			half = self.perimeter() / 2
			return 	sqrt(half*(half-self._a)*(half-self._b)*(half-self._c))
		
	def main():
		a,b,c = 3,4,5
		
		# 静态方法和类方法都是通过给类发消息来调用的
		if Triangle.is_valid(a,b,c):
			t = Triangle(a,b,c)
			#print(t.perimeter)
			# 也可以通过给类发消息来调用对象方法但是要传入接收消息的对象作为参数
			print(Triangle.perimeter(t))
			print(t.area())
			print(Triangle.area(t))
		else:
			print("无法构建三角形。")

	if __name__ == "__main__":
		main()
	
	```
+ 类方法：类方法的第一个参数约定名是cls，它代表当前类相关信息的对象(类本身是一个对象，也称类的元数据对象)，通过这个参数可以获取和类相关的信息并创建出类的对象。
	```python
	from time import time, localtime, sleep
	
	class Clock(object):
		"""数字时钟"""
		def __init__(self, hour=0, minute=0, second=0):
			self._hour = hour
			self._minute = minute
			self._second = second
		
		# 定义类方法
		@classmethod
		def now(cls):
			ctime = localtime(time())
			return cls(ctime.tm_hour, ctime.tm_min, ctime.tm_sec)     # 将传入的类名赋给cls并创建对象
			
		def run(self):
			"""走字"""
			self._second += 1
			if self._second == 60:
				self._second = 0
				self._minute += 1
				if self._minute == 60:
					self._minute = 0
					self._hour += 1
					if self._hour == 24:
						self._hour = 0
						
		def show(self):
			"""显示时间"""
			return '%02d:%02d:%02d' %(self._hour, self._minute, self._second)

	def main():
		# 通过类方法创建对象并获取系统时间
		clock = Clock.now()
		while True:
			print(clock.show())
			sleep(1)
			clock.run()
			
	if __name__ == "__main__":
		main()

	```
	
### 抽象类	
> 抽象类指不能够利用该类创建对象，只能用于被其它类继承。如果一个类中存在抽象方法，那么这个类就不能实例化。
> python语言需要通过abc模块的ABCMeta元类和abstractmethod包装器来达到抽象类的效果。

	```python
	from abc import ABCMeta, abstractmethod
	
	class Pet(object, metaclass=ABCMeta):     # metaclass是可以用来创建类，其内部会返回一个类。ABCMeta是让类变成一个纯虚类，用abstractmethod修饰，子类必须实现某个方法。
		'''宠物'''
		
		def __init__(self,nickname):
			self._nickname = nickname
			
		@abstractmethod
		def make_voice(self):
			'''发出声音'''
			pass
			
	class Dog(Pet):
		'''狗'''
		
		def make_voice(self):
			print("%s: 汪汪汪..."%self._nickname)
		
	class Cat(Pet):
		'''猫'''
		
		def make_voice(self):
			print("%s: 喵...喵..."%self._nickname)
			
	def main():
		pets = [Dog("旺财"), Cat("凯迪"), Dog("大黄")]
		for pet in pets:
			pet.make_voice()
			
	if __name__ == "__main__":
		main()
	
	```
### 内建元类type
+ **元类的目的**
	+ 为了当**创建类**时能够自动地改变类
		+ 拦截类的创建
		+ 修改类的定义
		+ 返回修改之后的类
+ type可以接受一个类的描述作为参数，然后返回一个类
	```python
	type(类名, 父类的元组(针对继承的情况，可以为空), 包含属性的字典(名称和值))
	Foo = type('Foo', (), {'bar':True})
	FooChild = type('FooChild', (Foo,), {'echo_bar': echo_bar})
	
	type的三个参数分别是：
		name: 要生产的类名
		bases: 包含所有基类的tuple
		dict: 类的所有属性，是键值对的字典

	```
+ type类和object类的关系
	+ python中的所有类是type类的实例，但是元类是type的子类，object是type的实例，而type是object的子类

	
----

## 进程和线程和协程

### 进程
+ 定义：**系统进行资源分配和调度的基本单位**，是操作系统结构的基础，可以理解为执行中的程序就是进程，每个进程都有自己的内存空间，进程之间的内存是独立的。一个进程由多个线程组成。程序运行时系统就会创建一个进程，并为它分配资源，然后把该进程放入进程就绪队列，进程调度器选中它的时候就会为它分配CPU时间，程序开始真正运行。



### 线程
+ 定义：**程序执行时的最小单位**，是进程的一个执行流，是**CPU调度和分派的基本单位**，线程之间共享进程的所有资源，每个线程有自己的堆栈和局部变量。线程由CPU独立调度执行，在多CPU环境下允许多个线程同时运行。同样多线程也可以实现并发操作，每个请求分配一个线程来处理。
+ 子程序的概念：函数A调用B，B在执行过程中调用C，C执行完毕返回，B执行完毕返回，最后A执行完毕。
	+ 调用顺序是确定的
+ 存在问题：线程切换存在开销

### 协程
+ 定义：**协程是**一种用户态的轻量级**线程**。如果说多进程对应多CPU，多线程对应多核CPU，那么协程则是在充分挖掘不断提高性能的单核CPU的潜力。
+ 跟协程不同，它是在一个子程序中中断，去执行其它子程序，可以利用generator来实现，比如yield和next
+ 解决问题：协程中切换不需要开销；不需要线程中的锁机制
```python
def consumer():
    r = ''
    while True:
        n = yield r
        if not n:
            return
        print('[CONSUMER] Consuming %s...' % n)
        r = '200 OK'

def produce(c):
    c.send(None)
    n = 0
    while n < 5:
        n = n + 1
        print('[PRODUCER] Producing %s...' % n)
        r = c.send(n)
        print('[PRODUCER] Consumer return: %s' % r)
    c.close()

c = consumer()
produce(c)
```

### 对比	
+ 进程是**资源分配**的最小单位，线程是**程序执行**的最小单位。相比较多线程，多进程更健壮，**多线程程序只要一个线程死掉，整个进程也死掉了**，而一个进程死掉并不会对另一个进程造成影响，因为进程有自己独立的地址空间。
+ 进程有自己的独立地址空间，每启动一个进程，系统就会为它分配地址空间，建立数据表来维护代码段、堆栈段和数据段，这种操作非常昂贵。而线程是共享进程中的数据，使用相同的地址空间，因此CPU切换一个线程的花费远比进程要小很多，同时创建一个线程的开销也比进程要小很多。
+ 线程之间的通信更方便，同一进程下的线程共享全局变量、静态变量等数据，而进程之间的通信需要以通信的方式(IPC)进行。不过如何处理好同步与互斥是编写多线程程序的难点。
+ 与多线程相比，由于不用线程之间的切换，协程的执行效率高。因为只有一个线程，也不存在同时写变量冲突，在协程中控制共享资源不需要线程锁。

### GIL锁
> python全局解释器锁，是一种互斥锁，仅允许一个线程持有python解释器的控制权，保证任何时间只能有一个线程处于执行状态。
+ 为什么需要GIL锁？
	+ python的内存管理：创建的对象具有引用计数变量，用于跟踪指向该对象的引用数，当计数为0时，释放该对象占用的内存。
		+ 下面的{}被引用了3次
		```python
		a = {}
        b = a
        c = b
		```
	+ 存在问题：当多个线程同时对一个值进行增加或减少操作时，可能会导致内存泄漏。
+ GIL锁：由于只存在一个锁，可以解决内存泄漏问题，同时不会产生死锁；
+ GIL锁的问题：
  	+ 对于cpu密集程序来说，由于只有1个线程获取锁，所以多线程无法很好的利用cpu并发处理能力；
  		+ 这个问题可以引入多进程+协程来解决
	+ 对于IO密集程序来说，GIL锁不会带来太大的问题
----

### 闭包
> 定义：一个返回值是函数的函数
```python
import time
def runtime():
	def now_time():
		print(time.time())
	return now_time

f = runtime()
f()

```

----

### python中拼接字符串+和join的区别
> 由于字符串是不可变对象，所以使用+会重新开辟新的内存空间，每拼接一次就要重新申请一次，效率低；
> join会进行一次性内存申请，运行效率较高。


