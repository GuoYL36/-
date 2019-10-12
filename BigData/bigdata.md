
## HDFS
+ 块的设计：太小会增加寻址开销，太大会使得资源浪费(使得MapReduce就在一两个任务上运行，不满足并行计算的宗旨)。
+ 名称节点
	+ FsImage：保存系统文件数
		+ 文件复制等级
		+ 块大小以及组成文件的块：不会具体记录块的存储位置，块的存储位置信息储存在内存中
		+ 修改和访问时间
		+ 访问权限
	+ EditLog: 记录对数据进行的诸如创建、删除、重命名等操作

## HBase
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
    + _ ：通配符
        + a._1：表示a中第一个元素

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






## SPARK
+ 分布式：就是一个RDD会被partition到不同node上
+ 迭代式计算：存在多个stage，某个stage执行得到的结果会传入下一stage继续执行。
+ 高度容错性：如果某个node上的数据发生丢失，那么它会从之前的计算节点上重新获取数据进行计算。
+ Job、Stage、Task
    + Job：提交给spark的任务
	+ Stage：每个Job的处理过程要分为几个步骤
	+ Task：运行的最小单位，每个Job的处理过程分为几次task
	+ 关系：Job ——> 一个或多个stage ——> 一个或多个task
		

## RDD：弹性分布式数据集（Resilient distributed datasets）
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




## DataFrame
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













     
