
#### 学习的时候按照架构图学更容易明白


## HDFS
+ 块的设计：太小会增加寻址开销，太大会使得资源浪费(使得MapReduce就在一两个任务上运行，不满足并行计算的宗旨)。
+ 名称节点
	+ FsImage：保存系统文件数
		+ 文件复制等级
		+ 块大小以及组成文件的块：不会具体记录块的存储位置，块的存储位置信息储存在内存中
		+ 修改和访问时间
		+ 访问权限
	+ EditLog: 记录对数据进行的诸如创建、删除、重命名等操作

### Hadoop基本命令
+ 初始化**（第一次使用）**：保证一致性
    + 找到配置里的tmp目录并删除
    + 重新进行格式化：hadoop name -format
+ 开启hadoop集群：/sbin/start-dfs.sh
+ 查看可执行指令：hadoop fs
+ 查看hdfs目录文件：hadoop fs -ls 目录名
+ 创建hdfs目录：hadoop fs -mkdir 目录名
+ 上传本地文件：hadoop fs -put 源文件 目的文件
+ 上传本地文件，限定源路径是一个本地文件：hadoop fs -copyFromLocal 源文件 目的路径
+ 下载hdfs文件，限定目标路径是一个本地文件：hadoop fs -copyToLocal 源文件 目的路径
+ 复制文件：hadoop fs -cp 源文件 目的文件
+ 递归删除目录：hadoop fs -rmr 目录名
+ 查看文件：hadoop fs -cat 文件名
+ hdfs中文件的复制：hadoop fs -cp 源文件 目的文件
+ 从hdfs下载文件到本地linux系统：hadoop fs -get [-ignorecrc] [-crc] hdfs的源文件 本地文件
+ 将源文件输出为文本格式，允许的格式为zip和TextRecordInputStream：hadoop fs -text 源文件
+ 从hdfs上复制文件到另一个hdfs上：hadoop distcp hdfs://hdfsClusterForML:9000/usr/spark/zhongyuan hdfs://192.168.1.178:9000/usr/spark

----
## MapReduce
+ 体系结构
    + Client(客户端)
        + 通过Client可以提交用户编写的应用程序到JobTracker端
        + 这些Client用户可以通过它提供的一些接口去查看当前提交作业的运行状态
    + JobTracker(作业跟踪器)
        + 负责资源的监控和作业的调度
        + 监控底层的其它的TaskTracker以及当前运行的Job的健康状况
        + 一旦探测到失败的情况就把这个任务转移到其它节点继续执行跟踪任务执行进度和资源使用量
    + TaskTracker(任务调度器)
        + 执行具体的相关任务一般接收JobTracker发送过来的命令
        + 把一些自己的资源使用情况，以及任务的运行进度通过心跳(heartbeat)的方式发送给JobTracker
        + slot概念：资源调度的单位，map类型的slot用于map任务；reduce类型的slot用于reduce任务
    + Task(任务)
        + map任务
        + reduce任务

+ 工作流程
    + 输入(split的各种分片，逻辑上的切分) ——> Map任务 ——> Shuffle(分区、排序、合并等) ——> Reduce任务 ——> 输出
    + split分成的片与block的区别：分片大小为hdfs的block大小；分片的数量等于map任务的数量。
    + Reduce任务的数量
        + 最优的Reduce任务个数取决于集群中可用的Reduce任务槽(slot)的数目
        + 通常设置比Reduce任务槽数目稍微小一些的Reduce任务个数(这样做可以预留一些系统资源处理可能发生的错误)
    + shuffle：当Map任务的输出在**缓存**中发生溢写时，数据会进行分区、排序以及合并并写到磁盘中，然后对磁盘文件归并，最后Reduce任务取走相应的数据并进行归并，最后在执行Reduce任务。
        + Map端的Shuffle过程：1、输入数据和执行Map任务；2、写入缓存(如果写到磁盘，开销太大)；3、溢写(分区、排序、合并)：存在溢写比,比如当使用80%的缓存时开始溢写，这样就能保证Map任务一直进行；4、文件归并
            + Map端shuffle过程(未使用合并): 在一个分区中，<a,<1>>+<a,<1>>——><a,<1,1>>
            + Map端shuffle过程(使用合并): 在一个分区中，<a,<1>>+<a,<1>>——><a,2>            
        + Reduce端的Shuffle过程：1、得到Map任务后的数据进入到缓存中；2、归并数据(把所有分区中Map的结果进行合并)；3、把数据输入给Reduce任务进行处理。
    + 用户程序部署到Master和Worker机器上，然后Master分配Map任务到一部分Worker，分配Reduce任务到另一部分Worker，最后由Worker执行具体的任务，中间文件存放于本地磁盘，不会存到分布式文件系统中。
    
    
## HBase
+ 单机模式下启动hbase
    + 启动hbase：./bin/start-hbash.sh
    + 进入shell命令操作Hbase数据库：./bin/hbase shell
    + 关闭hbase：./bin/stop-hbase.sh
+ 伪分布模式下启动hbase
    + 启动hdfs：./sbin/start-dfs.sh
    + 启动hbase：./bin/stop-hbase.sh
    + 进入shell命令操作Hbase数据库：./bin/hbase shell
    + 关闭hbase：./bin/stop-hbase.sh
    + 关闭hdfs：./sbin/stop-dfs.sh
+ 目的：用来存储非结构化和半结构化的松散数据。通过水平扩展方式(面向列的存储)允许数千台服务器来存储海量数据文件
+ 底层技术
	+ 文件存储系统：HDFS
	+ 海量数据处理：HDFS MapReduce
	+ 协同管理服务：Zookeeper
+ HBase诞生的原因
	+ Hadoop主要解决大规模数据**离线批量处理**，没有办法满足大数据实时处理需求；
	+ 传统关系型数据库的扩展能力有限：面向行的存储

+ HBase数据模型：稀疏多维度的排序的映射表，每一个值都是未经解释的字符串，即Bytes数组。
    + 行键：一个行键可以有多个列限定符，多个列族
    + 列限定符：列
    + 列族：由过个列限定符构成。HBase由多个列族构成。
        + 支持动态扩展
        + 保留旧的版本
    + 时间戳：一个行键和列限定符确定一个单元格，而一个单元格里包含多个时间戳

+ 数据坐标概念
    + HBase对数据的定位
        + 采用**四维坐标**来定位
        + 必须确定行键、列族、列限定符、时间戳
        + 可认为是键值数据库
    + 传统关系数据库的定位
        + 只要通过一个行一个列**两个维度**确定一个数据
        + Excel表格就类似于关系数据库
        
+ 面向行的存储和面向列的存储对比
    + 面向行的存储：适用于事务操作型
        + 优点
            + 对于传统的事务性操作，需要每次插入一条数据的时候，会把这条购物记录的各项信息都存入数据库，包括商品名称和价格
            + 在OLTP系统中，每一次都生成一个完整的记录
        + 缺点
            + 年龄的字段：对于行式存储来说，为了取出年龄这一列数据，先扫描数据库的第一行，然后这行中的年龄字段取出来，再去扫描第二行
            + 分析年龄分布特征、分析性别特征等等，都是针对一个列去分析，而面向行的存储需要扫描每一行去取出相应地列，代价很高
    + 面向列的存储：适用于数据分析
        + 优点
            + 很高的数据压缩率        


### HBase的实现原理
> 一个HBase表按照**行键字典序**被划分成多个Region，随着数据增多，一个Region会分裂成多个新的Region。
> 一个Region的大小从100M~200M变为1G到2G，取决于单台服务器的有效处理能力。每个Region服务器大概可以存储10到1000个Region。
> 同一个Region不会拆分到不同的Region上。

+ HBase的功能组件
    + 库函数：用于链接每个客户端
    + Master服务器：充当管家的作用
        + 分区信息进行维护和管理
        + 维护了一个Region服务器列表
        + 整个集群当中有哪些Region服务器在工作
        + 负责对Region进行分配
        + 负载平衡
    + Region服务器：负责存储不同的Region服务器
        + 客户端从Region服务器访问数据：客户端不依赖Master去获取位置信息
    
+ HBase中Region到底被存到哪里？**三层结构实现Region的寻址和定位**
    + 构建一个元数据表，假设该表只有两列，第一列是Region的id，第二列是Region服务器的id
    + HBase最开始构建时有一个映射表，这个映射表被称作为.META.表，用于存储元数据
    + 三层结构：Zookeeper文件 ——> -ROOT-表 ——> .META.表(多个) ——> 用户数据表(多个)
        + Zookeeper文件：记录-ROOT-表的位置信息
        + -ROOT-表：记录了.META.表的Region位置信息，**-ROOT-表只能有一个Region**，通过-ROOT-表就可以访问.META表中的数据
        + .META.表：记录了用户数据表的Region位置信息，**.META.表可以有多个Region**，保存了HBase中所有用户数据表的Region位置信息，会保存在内存中。
+ 客户端访问数据时的“三级寻址”：Zookeeper文件 ——> -ROOT-表 ——> .META.表(多个) ——> 用户数据表(多个)
    + 为了加速寻址，客户端会缓存位置信息
    + 缓存失效问题如何解决？
        + **“惰性”缓存**：第一次缓存后，后面直接用这个缓存，当发现这个缓存异常时，再重新进行“三级寻址”
        
### HBase的运行机制
+ HBase的系统架构
    + 第一层(最上层)
        + 客户端：范文HBase的接口
        + Zookeeper：提供配置维护、域名服务、分布式同步服务，大量用于分布式系统；实现协同管理服务；提供管家的功能(维护和管理整个HBase集群)
        + Master：对表增删改查；负责不同Region服务器的负载均衡；负责调整分裂、合并后Region的分布；负责重新分配故障、失效的Region服务器
    + 第二层：HBase的Region服务器：负责用户数据的存储和管理
    + 第三层：HDFS
    + 第四层(最底层)：Hadoop的数据节点

+ Region服务器的工作原理
    + 负责用户数据的存储和管理
    + 一个Region服务器里的所有Region共用一个HLog

+ 用户读写数据过程
    + 写数据
        + 分配到相应的Region服务器，数据写入MemStore缓存，写入MemStore缓存之前，为了保证数据的安全和可恢复性，必须先写HLog日志，只有HLog日志被完整写到磁盘当中时，MemStore缓存才会写入磁盘，最后返回调用数据的接口。
    + 读数据
        + 因为最新的数据在MemStore缓存中，所以先到相应的Region服务器访问MemStore缓存
        + 如果MemStore缓存找不到数据，再到磁盘的StoreFile找相关数据
    + 缓存刷新
        + 系统会周期性地把MemStore缓存里的内容刷写到磁盘的StoreFile文件中，清空缓存，并在HLog日志文件里写入一个标记
        + 每次刷写都生成一个新的StoreFile文件，因此，每个Store包含多个StoreFile文件
        + 每个Region服务器都有**一个**自己的HLog文件，每次启动都检查该文件，确认最近一次执行缓存刷新操作之后是否发生新的写入操作；如果发现更新，则先写入MemStore，再刷写到StoreFile，最后删除旧的HLog文件，开始为用户提供服务。

+ Store：包含多个StoreFile文件
    + StoreFile文件的合并：每次刷写都会产生新的StoreFile，多次刷写就会有多个StoreFile，影响查找速度，这时多个文件会合并成一个大的StoreFile文件
    + 合并和分裂过程
        + 4个64M的StoreFile合并成1个256M的StoreFile，然后1个256M的StoreFile分裂为2个128M的StoreFile
    

+ HLog：Region服务器的日志文件
    + 为什么设置？
        + 因为集群会出现故障。Zookeeper用来检测故障，Master来处理故障。
    + 一个Region服务器里的所有Region共用一个HLog，一个HLog负责一个10到1000个Region
    + 为什么一个Region服务器里只有一个HLog日志文件？
        + **提高对表的写操作性能**因为如果每个Region都配置一个HLog日志文件，那么一个Region服务器上就有很多个HLog日志文件，在写数据时，要打开所有HLog文件进行写操作，很费时；而只设置一个HLog日志文件的话，只需要打开一个文件进行写，有助于提升写操作性能。
        + 只有一个HLog日志文件，假如发生故障，那么对这个文件进行Region拆分也很耗时？
            + 由于Region服务器集群出现故障的几率小
    
### HBase的应用方案
+ 性能优化方法
    + 时间靠近的数据都存放在一起
    + 用系统最大的整型值减去时间戳：Long.MAX_VALUE - timestamp作为行键

+ 提升读写性能
    + 设置HColumnDescriptor.setInMemory选项为true，把相关的表放到Region服务器的缓存中，根据需要决定是否放入缓存
    + 设置最大版本数
    + 设置自动清除过期数据

+ 构建二级索引
    + 原生只支持对rowKey行键进行索引，不支持其它列构建索引
    + 优点：
        + 非侵入性：引擎构建在HBase之上，不对HBase进行任何改动，也不需要上层应用做任何妥协
    + 缺点：
        + 耗时变为双倍，对HBase集群的压力也是双倍的：每插入一条数据需要向索引表插入数据

+ 查询HBase检测性能
    + Master-status
    + Ganglia
    + OpenTSDB
    + Ambari
    
## scala
+ 各种符号
    + 参数名:_* ：将该参数当作参数序列处理
    + _ 通配符
        + a._1 表示a中第一个元素
        + 代表某一类型的默认值
            + 对于Int，它是0
            + 对于Double，它是0.0
            + 对于引用类型，它是null
    
    + 4种操作符：::, +:, :+, :::, ++
        + **::**被称为cons，意为构造，向队列的头部追加数据，创造新的列表。可用于pattern match。
        ```scala
        x::list  # 等价于list.::(x)，无论x是列表与否，都只将成为新生成列表的第一个元素
        ```
        + **:+**为用于在尾部追加元素
        + **+:**为用于在头部追加元素
        + **:::**方法**只能**用于**连接两个List类型**的集合。
        + **++**方法用于连接两个集合
        
    + 反斜杠\：scala.xml.Elem中的方法，用来解析XML
        + 提取XML中的子元素：val name: scala.xml.NodeSeq = node \ "name"
        + 提取二级元素：val name: scala.xml.NodeSeq = node \\ "name"
    + .head和.tail
        + .head为首个元素
        + .tail为剩余其它元素的序列
    + 可变长参数：int*、string*、...
    + 多行字符串—3个双引号""""""：可用于换行，让SQL结构清晰易于阅读
    
    
+ 字符串插值：将变量引用直接插入处理过的字面字符中
    + s字符串插值器：在任何字符串前加上s，就可以直接在字符串中使用变量或表达式。
    + f插值器：在任何字符串前加上f，就可以生成简单的格式化串，类型于printf
    + raw插值器：功能与s插值器一样，另外，raw插值器对字面值中的字符不做编码，比如"\n"不会转换成回车。
    + 高级用法：json格式，参考https://docs.scala-lang.org/zh-cn/overviews/core/string-interpolation.html#

+ 小括号和花括号
    + 函数只有一个参数时，两者可以互换；函数有两个或以上的参数时，只能使用小括号；
    + 调用单一参数函数时，虽然两者可以互换，但是小括号只接受单一的一行；花括号可以接受多行；
    + 调用一个单一参数函数时，如果参数本身是一个通过case语句实现的，偏函数只能使用花括号；
    + 非函数调用时，小括号用于界定表达式，花括号用于界定代码块。
        ```
        1：字面量
        (1)：表达式
        {1}：代码块
        ({1})：表达式里是一个语句块
        {(1)}：语句块里是一个表达式
        ```
+ implicit：隐式转换，对函数参数进行类型转换；对函数的调用者的类型进行转换。
    + 发生类型不匹配的函数调用时，会优先进行函数参数的类型转换，否则对函数的调用者的类型进行转换。
    + 使用方式 
        + 将方法或变量标记为implicit
        + 将方法的参数列表标记为implicit
        + 将类标记为implicit

+ scala和java的集合类型相互转换
    + 第一种方法，完成双向自动转型：import scala.collection.JavaConversions._
        + import scala.collection.convert.wrapAll._ //这个和引入collection.JavaConversions._ 没什么分别
        + import scala.collection.convert.wrapAsJava._  //单纯完成 Scala 到 Java 集合类型的隐式转换
        + import scala.collection.convert.wrapAsScala._ //只是完成 Java  到 Scala 集合的隐式转换
    + 第二种方法，引入 scala.collection.JavaConverters._, 显示调用asJava() 或 asScala() 方法完成转型。
    
+ sortBy和sortByKey
    + sortBy[K](f:(T)=>K, ascending:Boolean={}, numPartitions:Int={})
        + 返回类型：K
        + 接收参数类型：T，传入一个参数就可以按这个参数大小进行排序。
        + 排序方法：true为升序(默认)，false为降序。
        + 排序后RDD的分区个数：numPartitions，默认与排序前一致。
        + 原理：先调用keyBy函数生成key->value，然后利用sortBykey函数进行排序。
    
    + sortByKey(ascending:Boolean={}, numPartitions:Int={})
        + 针对于(key,value)的情况
    
+ map
    + map(_._2)：等价于map(t => t._2)，t是个2项以上的元组，即(1,2,3),(4,5,6),...;
    + map(_._2, _)：**网上说**等价于map(t => t._2, t)，返回元组的第二项及元组，假如元组是(1,2,3),(4,5,6),...，返回：2,(1,2,3)，5,(4,5,6)
    + map(t => (t._2, t))：上面写法在编译器里报错，改成这种写法没问题。

+ Option常用方法
    + getOrElse：获取元组中存在的元素或使用其默认值。
        + 强制规定返回值类型：getOrElse[Int]

+ apply
    + 什么情况下会调用apply?
        + **实例化**
            + 一般对象的实例化方法：A a=new A()，此时调用的是**this构造器**。
            + 如果有apply方法：A a=A()，此时调用的是A().apply()，返回的是**伴生对象**.**只要使用apply实例化的类，必有伴生对象**。**伴生对象中有apply方法，构造类的时候一般不用new**。
            + 为什么不用构造器？因为对于嵌套表达式(Array(Array("1","2"),Array("3","4")))，**省去new关键字方便很多**。
            + **Scala中很多类默认都是有伴生对象的**.
        + **查找属性**
            + 如果对象是集合，后面使用apply，则具有查找功能。
 
+ 拉链操作zip和zipWithIndex和zipWithUniqueId
    + zipWithIndex: 自动加索引，索引从0开始
    ```scala
    val name = Array("zhangsan","lisi","wangwu")
    name.zipWithIndex  // 输出：Array(("zhangsan",0),("lisi",1),("wangwu",2))
    
    // 将RDD中的元素和这个元素在RDD中的索引号组合成键/值对
    var rdd1 = sc.makeRDD(Seq("A","B","R","D","F"),2)
    rdd1.zipWithIndex().collect()  // 输出：Array[(String, Long)]=Array((A,0),(B,1),(R,2),(D,3),(F,4))
    ```
    + zip：合并两个集合成为一个二元组集合
    ```scala
    val name = Array("zhangsan","lisi","wangwu")
    val age = Array(18,19,20)
    val out = name zip age   //输出：Array(("zhangsan",18),("lisi",19),("wangwu",20))
    
    // 这样写会多打印一个()：out.foreach(println) // 改成这样: out.foreach{case(name, age)=> println(name,age)}
    ```
    + zipWithUniqueId：将RDD中元素和一个唯一ID组合成键值对，该唯一ID生成算法如下：每个分区中第一个元素的唯一ID值为**该分区索引号**；每个分区中第N个元素的唯一ID值为**前一个元素的唯一ID值+该RDD总的分区数**。
    ```scala
    var rdd1 = sc.makeRDD(Seq("A","B","C","D","E","F"),2)
    rdd1.zipWithUniqueId().collect()   //输出：Array((A,0),(B,2),(C,4),(D,1),(E,3),(F,5))
    //总分区数为2
    //第一个分区第一个元素A的ID为0，第二个分区第一个元素D的ID为1
    //第一个分区第二个元素B的ID为0+2=2，第二个分区第二个元素E的ID为1+2=3
    //第一个分区第三个元素C的ID为2+2=4，第二个分区第三个元素F的ID为3+2=5
    ```
+ Any类型转换为其它类型: asInstanceOf[Types]
+ 过程：函数体在花括号中，没有前面的=号，返回类型是Unit。
+ 异常
    + scala没有“受检"异常(编译期被检查)，即不需要声明函数或方法可能会抛出某种异常；
    + throw表达式有特殊的类型Nothing
        + 如果在if/else表达式中有一个分支是Nothing，那么if/else表达式的类型就是另一个分支。
        ```scala
        if(x >= 0){
            sqrt(x)
        }else throw new IllegalArgumentException("x should not be negative.")
        ```
    + 
+ split特殊符号
    + 关于点的问题是用string.split("[.]") 解决。
    + 关于竖线的问题用 string.split("\\|")解决。
    + 关于星号的问题用 string.split("\\*")解决。
    + 关于斜线的问题用 sring.split("\\\\")解决。
    + 关于中括号的问题用 sring.split("\\[\\]")解决。


## SPARK
+ 分布式：就是一个RDD会被partition到不同node上
+ 迭代式计算：存在多个stage，某个stage执行得到的结果会传入下一stage继续执行。
+ 高度容错性：如果某个node上的数据发生丢失，那么它会从之前的计算节点上重新获取数据进行计算。
+ Job、Stage、Task
    + Job：提交给spark的任务
	+ Stage：每个Job的处理过程要分为几个步骤
	+ Task：运行的最小单位，每个Job的处理过程分为几次task
	+ 关系：Job ——> 一个或多个stage ——> 一个或多个task


### RDD：弹性分布式数据集（Resilient distributed datasets）
> 分布式的Java对象的集合
+ 弹性：RDD进行partition，每个block最大只能存放128M，被分片到node的数据先存放于内存中，如果数据大小大于block大小，则存放于硬盘中，基于内存和硬盘的存取方式就是弹性。


+ RDD依赖关系只需记录这种粗粒度的转换操作，不需记录具体的数据和各种细粒度操作的日志。粗粒度就是针对批量数据集，细粒度就是数据集里某个特征等等。

+ RDD的两种生成方式
	```scala
	val rdd1 = sc.textFile("hdfs://node01：9000/ww")  //读取hdfs中节点9000下的文件ww
	val rdd2 = sc.parallelize(Array(1,2,3,4,5,6,7,8)) // 还可以传入List
	```

+ 两种算子：
	+ tranformation算子：只用来描述数据转换的依赖关系(DAG，有向无环图)，一种计算逻辑。这也是SPARK快的原因，输入按照逻辑关系得到输出，不需要保存中间结果。
		+ 变换算子：不会触发提交作业，完成作业中间过程处理(RDD -> RDD)，产生新的数据集。
			Value数据类型的变换算子：处理的数据项是Value型的数据；非Key-Value类型的RDD分区的值是None。
			Key-Value数据类型的变换算子：处理的数据项是Key-Value型的数据；只有Key-Value类型的RDD才有分区的。
			20种操作：map、filter、flatMap、mapPartitions、mapPartitionsWithIndex、sample、union、intersection、distinct、groupByKey、reduceByKey、aggregateByKey、sortByKey、join、cogroup、cartesian、pipe、coalesce、repartition、repartitionAndSortWithinPartitions
		
		+ action算子：开始按照transformation算子里的依赖关系开始运行，并输出结果
			行动算子：触发SparkContext提交job作业，进行计算和返回结果。
			12种操作：reduce、collect、show、count、first、takeSample、take、takeOrdered、saveAsTextFile、saveAsSequenceFile、saveAsObjectFile、countByKey、foreach.

+ 打印RDD元素的问题
    + 在本地模式下，可以将RDD中所有元素打印到屏幕上；
    + 在集群模式下，worker节点上打印是输出到worker节点的stdout上，而不是任务控制节点Driver上；为了解决这个问题，可以使用RDD.collect()方法将各个worker上的RDD元素打印到Driver上，但是当数据量大时，会导致内存溢出，因此要注意在Driver中最好只打印部分元素，而不是所有元素。    

+ 读取文件并转换成DataFrame格式
    ```scala
    import org.apache.spark.sql.SparkSession  // datasets和DataFrame的主要spark程序入口
    
    val spark = SparkSession.builder().getOrCreate()
    
    import spark.implicits._  // 使得支持RDDs转换为DataFrames及供后续sql操作
    
    
    ```
    
+ org.apache.spark.ml.feature.RFormula
    + ~ 分隔target列和其它列；
    + + 合并变量，+0意为移除截距；
    + - 移除变量，-1意为移除截距；
    + : 变量相交，（数字型值相乘，类别型值二值化）；
    + . 除了target的所有列
    + 对于字符型数据，RFormula使用StringIndexer转化为Double类型，然后执行OneHotEncoderEstimator
        StringIndexer：默认情况下是按照数据出现频率对数据进行排序，然后取其索引生成新的列。
    + 对于数字型数据，RFormula将其转为double类型。
    ```spark
    | id | country | hour | clicked |
    | 7  |   "US"  |  18  |   1.0   |
    | 8  |   "US"  |  18  |   1.0   |
    | 9  |   "US"  |  18  |   1.0   |
    
    (1)输入：clicked ~ country
    结果：先StringIndexer，再OneHotEncoderEstimator
    | label | features    |
    |  1.0  |  (2,[],[])  |
    |  0.0  |   [1.0,0.0] |
    |  0.0  |   [0.0,1.0] |
    
    (2)输入：clicked ~ country + hour
    结果：
    | label | features         |
    |  1.0  |  [0.0,0.0,18.0]  |
    |  0.0  |  [1.0,0.0,12.0]  |
    |  0.0  |  [0.0,1.0,15.0]  |    
    
    (3)输入：clicked ~ . - id
    结果：
    | label | features         |
    |  1.0  |  [0.0,0.0,18.0]  |
    |  0.0  |  [1.0,0.0,12.0]  |
    |  0.0  |  [0.0,1.0,15.0]  | 
    ```

+ org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
    + VectorAssembler：将多个数值列按顺序汇总成一个向量列
    + VectorIndexer：将一个向量列进行特征索引，即对每一列不同的值从小到大进行排序，然后取其索引当作新的向量列
    + setMaxCategories()方法指定该特征的取值超过多少会被视为连续特征。对连续特征不做处理，对离散特征进行索引。
    
+ spark中机器学习的特征数据类型
    + 向量类型：本地向量、分布式向量
    + 矩阵类型：本地矩阵、分布式矩阵
    + 传入机器学习模型的数据必须是vector型，vector长度就是机器学习模型里的numFeatures值。
    
+ spark中查看最优模型中各参数：
    ```Scala
        val cvModel = cv.fit(data)
        val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
        // stage(2)表示pipeline中model位于setStages的第3个参数
        val lrModel = bestModel.stage(2).asInstanceOf[LogisticRegressionModel]
        // 查看特征数量
        lrModel.numFeatures
    ```

+ spark中createOrReplaceTempView和createOrReplaceTempView

    + Spark Application
        + 针对单个批处理作业
        + 多个job通过session交互式
        + 不断满足请求的，长期存在的server
        + 一个Spark job可以包含多个map和reduce
        + Spark Application可以包含多个session实例
        
    + Spark Session
        + SparkSession与Spark应用程序相关联：

        + session 是两个或更多实体之间的交互媒介
        + 在Spark 2.0中，你可以使用SparkSession创建
        + 可以在不创建SparkConf，SparkContext或SQLContext的情况下创建SparkSession（它们封装在SparkSession中）

    + createOrReplaceTempView：创建或替换临时视图，该视图的生命周期是属于创建该数据集的当前SparkSession，如果新建SparkSession，则访问不到；
    + createGlobalTempView：创建全局临时视图，该视图的生命周期属于spark application

+ org.apache.spark.rdd
    + PartitionCoalescer.coalesce: 合并RDD的分区 

+ org.apache.spark.mllib.evaluation
    + BinaryClassificationMetrics：第一个参数为预测列，第二个参数为实际标签列

+ spark sql中agg函数
    + agg(exprs:column*)：返回DataFrame类型，计算求值；例如：df.agg(max("age"),avg("salary"))
    + agg(exprs:Map[String,String])：返回dataframe类型，计算求值；例如：df.agg(Map("age"->"max","salary"->"avg"))
    + agg(aggExpr: (String,String), aggExprs: (String,String)*)：返回dataframe类型，计算求值；例如：df.agg(Map("age"->"max","salary"->"avg"))
    
+ jdbc连接数据库
    + 首先在pom.xml中引入mysql-connector-java依赖
    ```scala
    import org.apache.spark.{SparkConf,SparkContext}
    import org.apache.spark.sql.{SQLContext, }
    
    val conf = new SparkConf().setMaster("local").setAppName("test")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    val jdbcDF = sqlContext.read.format("jdbc")
            .option("url","jdbc:mysql://localhost:3306/test?serverTimezone=GMT") // 数据库地址和名字、时区
            .option("driver","com.mysql.cj.jdbc.Driver")
            .option("dbtable","tb2")  // 数据库表名
            .option("user","root")  // 数据库用户名
            .option("password","1234")  // 数据库密码
            .load()
    jdbcDF.show()
    
    ```

### DataFrame
> 分布式的Row对象的集合，即：Dataset[Row]，可以自定义case class来获取每一行的信息
> 一般通过构建dataframe视图，利用spark.sql来对列求取最大值、最小值、平均值等;
> spark中有单独的column实例，但是一般对列进行运算还是会基于dataframe利用spark.sql来使用

### Row
+ 表示一行数据，可以通过索引获取元素值
```scala
    
    val a: DataFrame = Seq((1,2,3)).toDF("a","b","c")
    var b: Array[Row] = a.collect()
    var c: Row = b(0)
    var d: Integer = c.getInt(0)  // 输出为1
```

+ 取列操作：有的时候一个解析不了可以更换
    + df(列名)
    + column(列名)：位于org.apache.spark.sql.functions中
    + col(列名)：位于org.apache.spark.sql.functions中
    + $列名
    + 批量转换列格式
    ```scala
    import org.apache.spark.sql.functions._
    import org.apache.spark.sql.types._
    // 第一种方法
    val columns = df.columns
    var var_df = df
    for(column<-columns){
        var_df = var_df.withColumn(column, col(column).cast(DoubleType))
    }
    
    // 第二种方法
    val columns = df.columns
    val newTypeColumns = columns.map(column=>col(column).cast(DoubleType))
    df.select(newTypeColumns:_*)
    ```



### Dataset、dataframe和RDD的区别
> https://www.cnblogs.com/Transkai/p/11360603.html
+ 数据格式
    + RDD
    
    ||
    | :---------: |
    | 1, 张三, 23 |
    | 2, 李四, 35 |
    + DataFrame
    
    | ID:String | Name:String | Age:int |
    | :-------: |  :--------: | :-----: |
    |     1     |    张三     |    23   |
    |     2     |    李四     |    35   |
    + Dataset
        + 第一种 
        
          | value:String |
          | :----------: |
          | 1, 张三, 23  |
          | 2, 李四, 35  |
        + 第二种
        
          | value:People[age:bigint, id:bigint, name:String] |
          | :-----------------------------: |    
          |People(id=1, name="张三", age=23)|
          |People(id=2, name="李四", age=35)|

        
+ dataframe一般只针对列进行运算
+ RDD一般针对行进行运算
    ```scala
    val a: DataFrame = Seq((1,2,3)).toDF("a","b","c")
    val b: RDD[Row] = a.rdd
    val c：RDD[Any] = b.map(x=>x(0))  // 实际为RDD[(1,2,3)]
    val d: Array[Any] = c.collect()  // Array[(1,2,3)]
    
    // 打印b中每一行
    b.map(x=>x(0)).foreach(println)
    
    ```
+ 转换
    + DataFrame、DataSet转RDD
    ```scala
        val rdd1 = testDF.rdd  // DataFrame
        val rdd2 = testDS.rdd  // DataSet
    ```
    + RDD转DataFrame
    ```scala
        import spark.implicits._
        val testDF = rdd.map {line => (line._1, line._2)}.toDF("col1","col2")
    ```
    + RDD转DataSet
    ```scala
        import spark.implicits._
        case class Person(name:String, age:Int)
        val testDS = rdd.map {line => Person(line._1, line._2)}.toDS
        
    ```
    + DataSet转DataFrame
    ```scala
        import spark.implicits._
        val testDF = testDS.toDF
    ```
    + DataFrame转DataSet
    ```scala
        import spark.implicits._
        case class Person(name:String, age:Int)
        val testDS = testDF.as(Person)
    ```
### 各种算子
+ 转换操作算子
    + reduceByKey和groupByKey
        + reduceByKey
            + 使用方式
            ```Scala
            val a=sc.parallelize(Array("one","two","two")).map(word=>(word,1))      // ("one",1),("two",1),("two",1)
            val aReduce = a.reduceByKey(_+_)      // ("one",1),("two",2)
            ```
        + groupByKey
            + 使用方式
            ```Scala
            val a=sc.parallelize(Array("one","two","two")).map(word=>(word,1))      // ("one",1),("two",1),("two",1)
            val aReduce=a.groupByKey()      // ("one",[1]),("two",[1,1])
                        .map(x=>(x._1,x._2.sum)) // ("one",1),("two",2)
            ```
    + 区别
        + reduceByKey()会在shuffle之前对数据进行合并，也就是在map端进行合并操作；而groupByKey()在shuffle之后对数据进行合并，因此增加了IO开销；所以**从效率上，reduceByKey()优于groupByKey()**。
        

----

## 图计算
> 传统的图计算算法存在的典型问题：1、常常表现出比较差的内存访问局部性；2、针对单个顶点的处理工作过少；3、计算过程中伴随着并行度的改变 。
> 解决方案：1、为特定的图应用定制相应的分布式实现(通用性不好)；2、基于现有的分布式计算平台(MapReduce等)进行图计算，但是MapReduce是针对粗粒度(块)数据，而图计算是多次迭代、稀疏结构、细粒度数据，所以也不好；3、使用单机的图算法库(BGL等)，只能用于小数据；4、使用已有的并行图计算系统，如Parallel BGL，但是容错率低。**以上解决方案存在各种各样的缺陷**，所以产生了通用的图计算软件。
> 通用的图计算软件：1、基于遍历算法的、实时的图数据库(Neo4j、OrientDB等)；2、以图顶点为中心的、基于消息传递批处理的并行引擎(Giraph、Pregel等；3、这些都是基于**BSP模型**实现的，BSP(Bulk Synchronous Parallel Computing Model，整体同步并行计算模型，简称大同步模型)。4、BSP模型包含：通过网络连接起来的处理器；一系列的全局超步。
> 一个超步的结构(针对处理器)：1、局部计算组件；2、通讯组件；3、栅栏同步组件。

### 概念
+ Pregel
    + Pregel计算模型以**有向图**作为输入
    + 有向图的每个顶点都有一个String类型的顶点ID
    + 每个顶点都有一个可修改的用户自定义值与之关联
    + 每条有向边都和其源顶点关联，并记录了其目标顶点ID
    + 边上有一个可修改的用户自定义值与之关联
    + 在每个超步中，图中的所有顶点都会并行执行相同的用户自定义函数
    + 每个顶点可以接收前一个超步中发送给它的消息，修改其自身及其出射边的状态，并发送消息给其它顶点，甚至是修改整个图的拓扑结构
    + "边"并不是核心对象，在边上面不会运行相应的计算，只有顶点才会执行用户自定义函数进行相应计算
    + 传递消息的方法
        + 远程读取：work1把处理好的数据存在本地磁盘，通知work2远程读取work1的本地磁盘数据。会有高延迟。
        + 基于共享内存：可扩展性差
        + 基于消息传递模型(Pregel采用)：1、消息传递具有足够的表达能力；2、有助于提升系统整体性能。
+ 有向图和顶点
    + 消息存储于对应顶点的队列中

    + 顶点之间的消息传递


+ Pregel的计算过程
    + 由被称为“超步”的迭代组成
    + 每个超步中，每个顶点上都会并行执行用户自定义的函数
    + 一个算法什么时候结束由所有顶点的状态决定；
    + 在第0个超步，所有顶点处于活跃状态；
    + 一个顶点不需要执行进一步的计算就会把自己的状态设置为"停机"
    + 必须根据条件判断来决定是否将其显式唤醒进入活跃状态。
    + 超步t传递的消息，会在下一个超步执行；

+ 消息传递机制和Combiner
    + 每条消息都包含了消息值和需要达到的目标顶点ID；
    + 在一个超步S中，一个顶点可以发送任意数量的消息，这些消息将在下一个超步S+1中被其它顶点接收。
    + Pregel计算框架在消息发出去之前，Combiner可以将发往同一个顶点的多个整型值进行求和得到一个值，只需向外发送最后的结果，从而实现了由多个消息合并成一个消息，大大减少了传输和缓存的开销。
    + 默认情况下，Pregel计算框架并不会开启Combiner功能；当用户打算开启Combiner功能时，可以继承Combiner类并覆写虚函数Combine();此外，通常**只对满足交换律和结合律的操作采可以开启Combiner功能。**

+ Aggregator、拓扑改变和输入输出
    + Aggregator
        + Aggregator提供了一种全局通信监控和数据查看的机制，在一个超步S中，每一个顶点都可以向一个Aggregator提供一个数据，Pregel计算框架会对这些值进行聚合操作产生一个值，在下一个超步S+1中，图中所有顶点都可以看见这个值。
        + Aggregator定义一个"Sum" Aggrefator来统计每个顶点的出射边数量；
        + Aggregator实现全局协调功能；
    + 拓扑改变
        + 全局拓扑改变：因为会引发冲突，所以Pregel采用惰性协调机制，消息到达目标顶点，且被目标顶点要执行的时候，才会发生拓扑改变；
        + 本地拓扑改变：顶点或边的本地增减立即生效；
    
+ Pregel的执行过程和容错性
    + 大型图会被划分成多个分区，每个分区都包含了一部分顶点和以其为起点的边；
    + 一个顶点利用hash函数分配到对应的分区，可通过顶点ID判断属于哪个分区；
    + 执行过程的步骤
        + 选择集群中的多台机器执行图计算任务，一台为master，其它为worker；
        + master把图分成多个分区到多个worker，每个worker知道所有其它worker分配到的分区情况；
        + master会把用户输入划分成多个部分，并为每个worker分配用户输入的一部分。如果一个worker得到的是自己所分配的分区中的顶点，则立即更新数据结构，否则发送到其它分区；
        + master向每个worker发送指令开始运行一个超步，当worker完成工作会通知master，并把自己在下一超步处于“活跃”状态的顶点数量报告给master;
        + 计算过程结束后，master会通过每个worker进行持久化存储。
        
    + 容错性
        + 采用检查点机制来实现容错；
        + master周期性发送ping消息，每个worker收到后发送反馈信息给master，每个worker会保存一个或多个分区的状态信息；
        + master监测到worker发生故障后，会把失效worker所分配的分区重新分配到正常worker集合上。
+ Worker、Master和Aggregator
    + Worker
        + 信息：顶点当前值、出射边列表、消息队列、标志位
        + 执行过程中，信息保存在内存当中
        + 保存**一份**顶点值和边的值；标志位和消息队列是分开保存的，保存**两份**标志位和消息队列，两份一模一样。
            + 为啥保存两份标志位和消息队列
                + 一份用于当前超步(上一超步传递的消息)，另一份用于下一超步(当前超步其它顶点传递的消息)；标志位同理；
        + 消息传递到目标顶点
            + 如果目标顶点位于同一worker时，直接把消息放入到与目标顶点对应的输入消息队列中；
            + 如果不是位于同一worker时，先暂时缓存到本地，当缓存的消息数目到达预先设置的阈值时，才被批量异步发送到目标顶点的worker上(减少传输开销)；
    + Master
        + 主要协调worker执行各个任务；
        + 记录分区的数量，与顶点和边无关；
        + 内部运行了一个HTTP服务器来显示图计算过程的各种信息；
    + Aggregator
        + 在超步S中，每个worker会利用aggregator对本地分区中包含的所有顶点的值进行归约，得到一个本地的局部归约值；
        + 超步S结束时，所有worker会将局部归约值进行汇总得到全局值，提交给Master；
        + 下一超步S+1开始时，master会将Aggregator的全局值发送给每个worker;
### 算子        
+ reverse算子
    + 作用：把edge的方向反过来；
    ```scala
    val users: RDD[(VertexId, (String, String))] =
  sc.parallelize(Array((1L, ("a", "student")), (2L, ("b", "salesman")),
    (3L, ("c", "programmer")), (4L, ("d", "doctor")),
    (5L, ("e", "postman"))))

    val relationships: RDD[Edge[String]] =
      sc.parallelize(Array(Edge(1L, 2L, "customer"),Edge(3L, 2L, "customer"),
        Edge(3L, 4L, "patient"), Edge(5L, 4L, "patient"),
        Edge(3L, 4L, "friend"),   Edge(5L, 99L, "father")))

    val defaultUser = ("f", "none")

    val graph = Graph(users, relationships, defaultUser)
    
    graph.triplets.map(
          triplet => triplet.srcAttr._1 + " ——(" + triplet.attr + ")——> " + triplet.dstAttr._1
        ).collect.foreach(println(_))
    // 输出
    // a ——(customer)——> b
    // c ——(customer)——> b
    // c ——(patient)——> d
    // e ——(patient)——> d
    // c ——(friend)——> d
    // e ——(father)——> f
    
    // reverse算子
    val reverseGraph = graph.reverse
    reverseGraph.triplets.map(
      triplet => triplet.srcAttr._1 + " ——(" + triplet.attr + ")——> " + triplet.dstAttr._1
    ).collect.foreach(println(_))

    // 输出
    // b ——(customer)——> a
    // b ——(customer)——> c
    // d ——(patient)——> c
    // d ——(patient)——> e
    // d ——(friend)——> c
    // f ——(father)——> e
   
    ```
+ subgraph
    + 作用：取原来graph的子graph，获取子graph必须有条件过滤掉一部分数据；
    ```scala
    val subGraph = graph.subgraph(vpred = (id, attr) => attr._1 > "b")
    subGraph.triplets.map(
      triplet => triplet.srcAttr._1 + " ——(" + triplet.attr + ")——> " + triplet.dstAttr._1
    ).collect.foreach(println(_))

    // 输出
    // c ——(patient)——> d
    // e ——(patient)——> d
    // c ——(friend)——> d
    // e ——(father)——> f
   
    ```
+ mask算子
    + 作用：求当前graph和另外一个graph的交集
    ```scala
    val maskGraph = graph.mask(subGraph)
    maskGraph.triplets.map(
      triplet => triplet.srcAttr._1 + " ——(" + triplet.attr + ")——> " + triplet.dstAttr._1
    ).collect.foreach(println(_))

    // 输出：subGraph与graph的交集肯定是subGraph，因为subGraph是graph的子图。
    // c ——(patient)——> d
    // e ——(patient)——> d
    // c ——(friend)——> d
    // e ——(father)——> f
    
    ```
+ groupEdges
    + 原因：graphx处理的是多重图，即2个顶点之间可能有多条平行边；
    + 作用：将2个vertex之间所有edge进行合并；
    ```scala
    // 例子1：
    val combineGraph = graph
  .partitionBy(PartitionStrategy.EdgePartition1D)
  .groupEdges(merge = (e1, e2) => e1 + " and " + e2)

    combineGraph.triplets.map(
          triplet => triplet.srcAttr._1 + " ——(" + triplet.attr + ")——> " + triplet.dstAttr._1
        ).collect.foreach(println(_))
    
    // 输出：这里将平行边的元素用and连接起来了，这里要注意的是，使用groupEdges算子之前，必须先用一下partitionBy，因为它假设同一条边位于同一分区，不然不起作用的。
    // a ——(customer)——> b
    // c ——(customer)——> b
    // c ——(patient and friend)——> d
    // e ——(patient)——> d
    // e ——(father)——> f

    // 例子2
    // groupEdge ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
		// day09-02-edges.csv
		// 1,2,100,2014/12/1
		// 1,2,110,2014/12/11
		// 2,3,200,2014/12/21
		// 2,3,210,2014/12/2
		// 3,1,300,2014/12/3
		// 3,1,310,2014/12/31
		val edgeLines2: RDD[String] = sc.textFile("graphdata/day09-02-edges.csv")
		val e2:RDD[Edge[((Long, java.util.Date))]] = edgeLines2.map(line => {
				val cols = line.split(",")
				Edge(cols(0).toLong, cols(1).toLong, (cols(2).toLong, format.parse(cols(3))))
			})
 
		val graph2:Graph[(String, Long), (Long, java.util.Date)] = Graph(v, e2)
 
		// 使用groupEdges语句将edge中相同Id的数据进行合并
		val edgeGroupedGraph:Graph[(String, Long), (Long, java.util.Date)] = graph2.groupEdges(merge = (e1, e2) => (e1._1 + e2._1, if(e1._2.getTime < e2._2.getTime) e1._2 else e2._2))
 
		println("\n\n~~~~~~~~~ Confirm merged edges graph ")
		edgeGroupedGraph.edges.collect.foreach(println(_))
		// Edge(1,2,(210,Mon Dec 01 00:00:00 EST 2014))
		// Edge(2,3,(200,Sun Dec 21 00:00:00 EST 2014))
		// Edge(2,3,(210,Tue Dec 02 00:00:00 EST 2014))
		// Edge(3,1,(610,Wed Dec 03 00:00:00 EST 2014))

    
    ```

 
## Docker

+ 创建默认库：docker-machine create -d virtualbox 虚拟机名
+ 查看默认库：docker-machine ls
+ 删除默认库：docker-machine rm -f 虚拟机名
+ 开启docker的虚拟机：docker-machine start 虚拟机名
    





     
