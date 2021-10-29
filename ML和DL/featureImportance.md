## randomForest、GBDT、XGBoost的特征重要性如何计算？
> 参考：https://zhuanlan.zhihu.com/p/64759172
> 由于randomForest的特征重要性计算和GBDT差不多，下面只比较GBDT和XGBoost的计算过程

### GBDT模型的特征重要性计算
1、找到BaseGradientBoosting类，得到feature_importances_方法源码如下：
> url: https://github.com/scikit-learn/scikit-learn/blob/51a765acfa4c5d1ec05fc4b406968ad233c75162/sklearn/ensemble/gradient_boosting.py

```python
def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).
        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        self._check_initialized()

        total_sum = np.zeros
        # 这一步for循环从self.estimators_遍历每个回归树
        # 主类的estimators_方法用以输出gbdt训练过程中建立的决策树群
        for stage in self.estimators_:
        # 针对每个决策树子树，分别调用feature_importances_方法
        # 这说明决策树群中每个子树都对应一套特征重要性
        # 也说明BaseGradientBoosting的feature_importances_方法只是对决策树的feature_importances_进行变换，并非最原始的逻辑
            stage_sum = sum(tree.feature_importances_
                            for tree in stage) / len(stage)
        # 这里将每棵树的特征重要性数组合并加总成一个数组
            total_sum += stage_sum
        # 这里将合并后的数组值除以树的个数，可以看作是每个树平均的特征重要性情况
        importances = total_sum / len(self.estimators_)
        return importances
```
2、继续从tree中找feature_importances_源码，发现tree的feature_importances_来自于tree_.compute_feature_importances()方法：
> url: https://github.com/scikit-learn/scikit-learn/blob/51a765acfa4c5d1ec05fc4b406968ad233c75162/sklearn/tree/_tree.pyx

```cpython
cpdef compute_feature_importances(self, normalize=True):
        """Computes the importance of each feature (aka variable)."""
        cdef Node* left
        cdef Node* right
        cdef Node* nodes = self.nodes
        cdef Node* node = nodes
        cdef Node* end_node = node + self.node_count
        cdef double normalizer = 0.
        cdef np.ndarray[np.float64_t, ndim=1] importances
        importances = np.zeros((self.n_features,))
        cdef DOUBLE_t* importance_data = <DOUBLE_t*>importances.data

        with nogil:
        # 在计算impurity时，while和if用以过滤掉决策树中的根节点和叶子节点
            while node != end_node:
                if node.left_child != _TREE_LEAF:
                    left = &nodes[node.left_child]
                    right = &nodes[node.right_child]
        # 遍历每个节点，该节点对应分裂特征重要性统计量=分裂前impurity减去分裂后左右二叉树impurity之和
        # 计算impurity的差值时，每个impurity都乘以对应权重（分支的样本数）
        # 一个特征在树中可以被用来多次分裂，基于上一步的数据，等同于这里按照特征groupby后对其重要性统计量求和
                    importance_data[node.feature] += (
                        node.weighted_n_node_samples * node.impurity -
                        left.weighted_n_node_samples * left.impurity -
                        right.weighted_n_node_samples * right.impurity)
                node += 1
        # 每个特征的重要性统计量除以根节点样本总数，做了下平均
        importances /= nodes[0].weighted_n_node_samples
        # 各特征重要性统计量最重要转化成百分比度量
        # 该步将各特征统计量分别除以统计量总和，归一化操作
        # 转化成百分比就得到了我们看到的特征重要性
        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                # Avoid dividing by zero (e.g., when root is pure)
                importances /= normalizer

        return importances
```
3、GBDT是根据分裂前后节点的impurity减少量来评估特征重要性，那impurity的计算标准是什么？
答：在源码中_criterion.pyx脚本中criterion分裂标准有：
+ Entropy：熵，适用分类树
+ Gini：基尼系数，适用分类树
+ MSE：均方误差，适用回归树
+ MAE：平均绝对误差，适用回归树

由于GBDT中用的是回归树，所以impurity计算和节点分裂标准是MSE或MAE。
	


### XGBoost模型的特征重要性计算
> 源码位于https://github.com/zengfanxi/xgboost/blob/master/python-package/xgboost/core.py中get_score函数

+ get_score函数参数
	+ fmap：包含特征名称映射关系的txt文档；
	+ importance_type：importance的计算类型，有以下5个取值
		+ weight：权重（某特征在整个树群节点中出现的次数，出现越多，价值就越高）
		+ gain：（某特征在整个树群作为分裂节点的信息增益之和再除以某特征出现的频次）
		+ total_gain（同上，代码中有介绍，这里total_gain就是gain）
		+ cover：某个特征节点样本的二阶导数和除以该特征出现的频次
		+ total_cover

get_dump获取树结构信息：
```python
trees = bst.get_dump(with_stats=True)
for tree in trees:
    print(tree)
# 以下输出了2次迭代的决策树规则，规则内包含量特征名、gain和cover，
# 源码就是提取上述3个变量值进行计算特征重要性
[out]:
0:[inteval<1] yes=1,no=2,missing=1,gain=923.585938,cover=7672
	1:[limit<9850] yes=3,no=4,missing=3,gain=90.4335938,cover=6146.5
		3:leaf=-1.86464596,cover=5525.25
		4:leaf=-1.45520294,cover=621.25
	2:[days<3650] yes=5,no=6,missing=5,gain=164.527832,cover=1525.5
		5:leaf=-1.36227047,cover=598
		6:leaf=-0.688206792,cover=927.5

0:[days<7850] yes=1,no=2,missing=1,gain=528.337646,cover=4162.56592
	1:[frequency<4950] yes=3,no=4,missing=3,gain=64.1247559,cover=2678.6853
		3:leaf=-0.978122056,cover=1715.49646
		4:leaf=-0.653981686,cover=963.188965
	2:[interval<4] yes=5,no=6,missing=5,gain=179.725327,cover=1483.88074
		5:leaf=-0.256728679,cover=1280.68018
		6:leaf=0.753442943,cover=203.200531
```

源码：
```python
# 当重要性类型选择“weight”时：
        if importance_type == 'weight':
            # get_dump用以从模型中打印输出所有树的规则信息
            trees = self.get_dump(fmap, with_stats=False)

            fmap = {}
            # 以下for循环用以从所有树规则中提取出特征名称
            for tree in trees:
                for line in tree.split('\n'):
                    arr = line.split('[')
                    if len(arr) == 1:
                        continue
                    fid = arr[1].split(']')[0].split('<')[0]
                    # 以下语句利用if判断统计所有树规则中每个特征出现的频次
                    # 即每个特征在分裂时候被利用的次数
                    if fid not in fmap:
                        fmap[fid] = 1
                    else:
                        fmap[fid] += 1

            return fmap

        else:
        # 通过以下代码知道importance_type选择total_gain时其实就是gain；
        # 选择total_cover时也等同于cover
            average_over_splits = True
            if importance_type == 'total_gain':
                importance_type = 'gain'
                average_over_splits = False
            elif importance_type == 'total_cover':
                importance_type = 'cover'
                average_over_splits = False
            # 还是先打印输出所有树规则信息
            trees = self.get_dump(fmap, with_stats=True)

            importance_type += '='
            fmap = {}
            gmap = {}
            for tree in trees:
                for line in tree.split('\n'):
                    arr = line.split('[')
                    if len(arr) == 1:
                        continue

                    fid = arr[1].split(']')
                    # 该步计算g的时候利用了importance_type参数
                    # 当importance_type="gain"时，提取树规则中的gain值
                    # 当importance_type="cover"时，提取树规则中的cover值
                    g = float(fid[1].split(importance_type)[1].split(',')[0])
                    # fid和参数weight时候的fmap是一样的，其实都是从树规则中提取到的特征名称列表
                    fid = fid[0].split('<')[0]
                    # 这步if操作，涉及两个统计量
                    # 一是每个特征在所有树规则中出现的频次
                    # 二是每个特征在所有树节点上的信息增益之和
                    if fid not in fmap:
                        fmap[fid] = 1
                        gmap[fid] = g
                    else:
                        fmap[fid] += 1
                        gmap[fid] += g

            # 将特征重要性求均值得到最终的重要性统计量，具体方法是：
            # 当importance_type ="gain"：a特征重要性为a总的信息增益和除以树规则中a特征出现的总频次
            # 当importance_type = "cover时"，a特征重要性为a总的cover和除以树规则中a特征出现的总频次
            if average_over_splits:
                for fid in gmap:
                    gmap[fid] = gmap[fid] / fmap[fid]

```










