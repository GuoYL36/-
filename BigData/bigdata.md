
## HDFS
+ 块的设计：太小会增加寻址开销，太大会使得资源浪费(使得MapReduce就在一两个任务上运行，不满足并行计算的宗旨)。





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













     
