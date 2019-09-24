
## sql
+ 定义别名的方式
    + select 字段 as 别名;    // 更推荐使用 
    + select 字段 别名
    + select 别名=字段
    
    + as后面别名的写法有的加括号、单引号、双引号、以及没有引号
        + 对于别名中含有空格时，以下写法不报错
            + select student as '姓 名'
            + select student as [姓 名]
            + select student as "姓 名"
        + 对于别名中含有空格时，以下写法报错
            + select student as 姓 名





## Hive

+ over：开窗函数，和聚合函数的不同之处是，对于每个组返回多行，而聚合函数对于每个组只返回一行。
    + over (order by salary)：按照salary排序进行累计，order by是个默认的开窗函数
    + over (partition by salary)：按照salary分区

+ collect：将分组中的某列转为一个数组返回
    + collect_list：不会去重
    + collect_set：会去重








