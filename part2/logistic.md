# Logistic回归

logistic回归，也叫逻辑回归，虽然也叫回归，但是用来做分类的。

主要思想是：根据数据，对分类边界线建立回归公式，以此进行分类。



## 概念须知

### 回归

有一些数据点，我们要对这些数据点进行拟合（这条直线成为最佳拟合直线），这个拟合的过程就叫做回归（regression)。

### Sigmoid/Logistic 函数

公式：

$$
\sigma ( t ) = \frac { 1} { 1+ e ^ { - t } }
$$

函数图：

<img src="https://ws4.sinaimg.cn/large/006tNc79gy1frw5rwaisdj30i40gcgmc.jpg" width="30%" />



性质：

- x=0,y=0.5
- x增大，y逼近1
- x减小，y逼近0

想对Sigmoid 函数有更多了解，可以点开[此链接](https://www.desmos.com/calculator/bgontvxotm)跟此函数互动。

## 回归系数的确定

Sigmoid函数的输入记为z，由以下公式确定
$$
z = w _ { 0} x _ { 0} + w _ { 1} x _ { 1} + w _ { 2} x _ { 2} + \ldots + w _ { n } x _ { n }
$$
采用向量写法：
$$
z = w ^ { T } x
$$


向量**w**就是我们要求的最佳参数。我们使用梯度上升法（Gradient Ascent）

### 梯度上升法

#### 梯度概念：

```
向量 = 值 + 方向  
梯度 = 向量
梯度 = 梯度值 + 梯度方向
```

#### 梯度上升法思想

要找到某个函数的最大值，最好的方法是沿着该函数的梯度探寻。如果梯度记为 ▽ ，则函数 f(x, y) 的梯度由下式表示: 
$$
\nabla f ( x ,y ) = \left( \begin{array} { c } { \frac { \partial f ( x ,y ) } { \partial x } } \\ { \frac { \partial f ( x ,y ) } { \partial y } } \end{array} \right)
$$


这个梯度意味着要沿 x 的方向移动 $\frac { \partial f ( x ,y ) ) } { \partial x }$，沿 y 的方向移动$\frac { \partial f ( x ,y ) ) } { \partial y }$。其中，函数f(x, y) 必须要在待计算的点上有定义并且可微。下图是一个具体的例子。

<img src="https://ws3.sinaimg.cn/large/006tNc79gy1frx8ft4xkkj30wq0puwln.jpg" width="50%" />



上图展示的，梯度上升算法到达每个点后都会重新估计移动的方向。从 P0 开始，计算完该点的梯度，函数就根据梯度移动到下一点 P1。在 P1 点，梯度再次被重新计算，并沿着新的梯度方向移动到 P2 。如此循环迭代，直到满足停止条件。迭代过程中，梯度算子总是保证我们能选取到最佳的移动方向。

上图中的梯度上升算法沿梯度方向移动了一步。可以看到，梯度算子总是指向函数值增长最快的方向。这里所说的是移动方向，而未提到移动量的大小。该量值称为步长，记作 α 。用向量来表示的话，梯度上升算法的迭代公式如下:
$$
w : = w + \alpha \nabla _ { w } f ( w )
$$
问：有人会好奇为什么有些书籍上说的是梯度下降法（Gradient Decent）?

答： 其实这个两个方法在此情况下本质上是相同的。关键在于代价函数（cost function）或者叫目标函数（objective function）。如果目标函数是损失函数，那就是最小化损失函数来求函数的最小值，就用梯度下降。 如果目标函数是似然函数（Likelihood function），就是要最大化似然函数来求函数的最大值，那就用梯度上升。在逻辑回归中， 损失函数和似然函数无非就是互为正负关系。

只需要在迭代公式中的加法变成减法。因此，对应的公式可以写成:
$$
w : = w - \alpha \nabla _ { w } f ( w )
$$

### **局部最优现象 （Local Optima）**

![image-20180602231557913](https://ws1.sinaimg.cn/large/006tNc79gy1frx8r1ty4cj312u0jek4l.jpg)

上图表示参数 θ 与误差函数 J(θ) 的关系图 (这里的误差函数是损失函数，所以我们要最小化损失函数)，红色的部分是表示 J(θ) 有着比较高的取值，我们需要的是，能够让 J(θ) 的值尽量的低。也就是深蓝色的部分。θ0，θ1 表示 θ 向量的两个维度（此处的θ0，θ1是x0和x1的系数，也对应的是上文w0和w1）。

可能梯度下降的最终点并非是全局最小点，可能是一个局部最小点，如我们上图中的右边的梯度下降曲线，描述的是最终到达一个局部最小点，这是我们重新选择了一个初始点得到的。

看来我们这个算法将会在很大的程度上被**初始点**的选择影响而陷入局部最小点。

## 线性回归与logistic回归的区别

1）线性回归要求变量服从正态分布，logistic回归对变量分布没有要求。

2）线性回归要求因变量是连续性数值变量，而logistic回归要求因变量是分类型变量。

3）线性回归要求自变量和因变量呈线性关系，而logistic回归不要求自变量和因变量呈线性关系

4）logistic回归是分析因变量取某个值的概率与自变量的关系，而线性回归是直接分析因变量与自变量的关系

总之, logistic回归与[线性回归](http://baike.baidu.com/view/2362545.htm)实际上有很多相同之处，最大的区别就在于他们的[因变量](http://baike.baidu.com/view/324030.htm)不同，其他的基本都差不多，正是因为如此，这两种回归可以归于同一个家族，即广义线性模型（generalized linear model）。这一家族中的模型形式基本上都差不多，不同的就是[因变量](http://baike.baidu.com/view/324030.htm)不同，如果是连续的，就是多重线性回归，如果是[二项分布](http://baike.baidu.com/view/79831.htm)，就是logistic回归。logistic回归的[因变量](http://baike.baidu.com/view/324030.htm)可以是二分类的，也可以是多分类的，但是二分类的更为常用，也更加容易解释。所以实际中最为常用的就是二分类的logistic回归。



## 算法

```
每个回归系数初始化为 1
重复 R 次:
    计算整个数据集的梯度
    使用 步长 x 梯度 更新回归系数的向量
返回回归系数
```

## 开发流程

```
收集数据: 可以使用任何方法
准备数据: 由于需要进行距离计算，因此要求数据类型为数值型。另外，结构化数据格式则最佳
分析数据: 画出决策边界
训练算法: 使用梯度上升找到最佳参数
测试算法: 使用 Logistic 回归进行分类
使用算法: 对简单数据集中数据进行分类
```

## 优缺点

```
优点: 计算代价不高，易于理解和实现。
缺点: 容易欠拟合，分类精度可能不高。
适用数据类型: 数值型和标称型数据。
```

## 实践

[sklean](https://github.com/syfly007/algorithm/blob/master/MachineLearning/LogisticRegression/logistic_Sklearn.ipynb)

[tensorflow](https://github.com/syfly007/algorithm/blob/master/MachineLearning/LogisticRegression/logistic_tf.ipynb)

## 参考资料：

https://github.com/apachecn/MachineLearning/blob/master/docs/5.Logistic%E5%9B%9E%E5%BD%92.md

