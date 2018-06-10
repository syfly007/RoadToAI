# 引言

Logsitic回归和线性回归很像，区别是Logsitic回归虽然有”回归“两字，但却是做分类的。

# 基本原理

1.找到一个合适的预测函数h（Andrew Ng 的公开课中称为 hypothesis），用来表示我们的分类函数。这个过程很关键，我们需要知道数据大概“形式”，比如是线性还是非线性。

2.构造一个cost函数，表示预测函数h的输出和类别 y 之间的偏差。综合考虑所有训练数据的“损失”，将cost求和或者求平均，记为$J ( \theta )$函数，表示所有训练数据和实际类别的偏差。

3.显然，$J ( \theta )$越小，预测函数h越准确。所以要找到$J ( \theta )$的最小值。找函数的最小值有不同的方法，Logistic Regression 用的是梯度下降法(Gradient Descent)。

# 具体过程

## 构造预测函数 h

首先需要找到一个预测函数h，这个函数的输出必须是2个值，用来表示2个类别。这里，用到的是Logistic函数（或称Sigmoid函数）：
$$
g ( z ) = \frac { 1} { 1+ e ^ { - z } } \tag{1}
$$





对应函数图像是取值0～1之间的“S”形曲线：

**图1**

<img src="https://ws2.sinaimg.cn/large/006tKfTcgy1fs16pv6rmoj30i40gcq49.jpg" width="50%" />



**图2**

<img src="https://ws2.sinaimg.cn/large/006tKfTcgy1fs16la3uphj30ua0hyk4t.jpg" width="50%" />

**图3**

<img src="https://ws4.sinaimg.cn/large/006tKfTcgy1fs16najhmzj30tc0dy11w.jpg" width="50%" />

接下来我们要确定数据边界的类型。图2显然需要一个线性边界，图3需要一个非线性边界。后面，我们**只讨论线性边界**的情况。

对于线性边界，形式如下：
$$
\theta _ { 0} + \theta _ { 1} x _ { 1} + \ldots + \theta _ { n } x _ { n } = \sum _ { i = 0} ^ { n } \theta _ { i } x _ { i } = \theta ^ { T } x \tag{2}
$$
构造预测函数：
$$
h _ { \theta } ( x ) = g \left( \theta ^ { T } x \right) = \frac { 1} { 1+ e ^ { - \theta ^ { T}x } } \tag{3}
$$

$h _ { \theta } ( x )$表示结果为1的概率，因此，对于输入x，结果为1和0对概率分别为：
$$
\left.\begin{array} { l } { P ( y = 1| x ; \theta ) = h _ { \theta } ( x ) } \\ { P ( y = 0| x ; \theta ) = 1- h _ { \theta } ( x ) } \end{array} \tag{4} \right.
$$

## 构造Cost函数

Andrew Ng直接给出了Cost函数和$J ( \theta )$函数：

**Cost函数**
$$
\operatorname{Cost} \left( h _ { \theta } ( x ) ,y \right) = \left\{ \begin{aligned} - \log \left( h _ { \theta } ( x ) \right) & \text{ if } y = 1\\ - \log \left( 1- h _ { \theta } ( x ) \right) & \text{ if } y = 0\end{aligned} \tag{5}\right.
$$


**$J(\theta)$函数**
$$
\left.\begin{aligned} J ( \theta ) & = \frac { 1} { m } \sum _ { i = 1} ^ { m } \operatorname{Cost} \left( h _ { \theta } \left( x ^ { ( i ) } \right) ,y ^ { ( i ) } \right) \\ & = - \frac { 1} { m } \left[ \sum _ { i = 1} ^ { m } y ^ { ( i ) } \log h _ { \theta } \left( x ^ { ( i ) } \right) + \left( 1- y ^ { ( i ) } \right) \log \left( 1- h _ { \theta } \left( x ^ { ( i ) } \right) \right) \right] \end{aligned} \tag{6}\right.
$$


实际上Cost函数和$J ( \theta )$函数是从[最大似然估计](https://www.wikiwand.com/zh-hans/%E6%9C%80%E5%A4%A7%E4%BC%BC%E7%84%B6%E4%BC%B0%E8%AE%A1)推导得到的，以下为推导过程：



（4）式可以综合起来：
$$
P ( y | x ; \theta ) = \left( h _ { \theta } ( x ) \right) ^ { y } \left( 1- h _ { \theta } ( x ) \right) ^ { 1- y } \tag{7}
$$
取似然函数：
$$
\left.\begin{aligned} L ( \theta ) & = \prod _ { i = 1} ^ { m } P \left( y ^ { ( i ) } | x ^ { ( i ) } ; \theta \right) \\ & = \prod _ { i = 1} ^ { m } \left( h _ { \theta } \left( x ^ { ( i ) } \right) \right) ^ { y ^ { ( i) } } \left( 1- h _ { \theta } \left( x ^ { ( i ) } \right) \right) ^ { 1- y ^ { ( i) } } \end{aligned} \tag{8}\right.
$$
取对数似然函数：
$$
\left.\begin{aligned} l ( \theta ) & = \log L ( \theta ) \\ & = \sum _ { i = 1} ^ { m } \left( y ^ { ( i ) } \log h _ { \theta } \left( x ^ { ( i ) } \right) + \left( 1- y ^ { ( i ) } \right) \log \left( 1- h _ { \theta } \left( x ^ { ( i ) } \right) \right) \right) \end{aligned} \tag{9}\right.
$$
最大似然估计就是要求得使$l(\theta)$ 取最大值时的$\theta$，其实这里可以使用梯度上升法求解，求得的$\theta$就是要求的最佳参数。但是，在 Andrew Ng 的课程中将 $J(\theta)$取为(6)式，即:
$$
J ( \theta ) = - \frac { 1} { m } l ( \theta )\tag{10}
$$
因为乘了一个负的系数$- \frac { 1} { m }$，所以$J(\theta)$取最小时，$\theta$为要求的最佳参数。

## 梯度下降法求$J(\theta)$最小值

根据梯度下降法的$\theta$的更新过程：
$$
\theta _ { j } : = \theta _ { j } - \alpha \frac { \partial } { \partial \theta _ { j } } J ( \theta ) ,\quad ( j = 0\ldots n ) \tag{11}
$$

$$
\left.\begin{aligned} 
\frac { \partial } { \partial \theta _ { j } } J ( \theta ) 
& = - \frac { 1} { m } \sum _ { i = 1} ^ { m } \left( y ^ { ( i ) } \frac { 1} { h _ { \theta } \left( x ^ { ( i ) } \right) } \frac { \partial } { \partial \theta _ { j } } h _ { \theta } \left( x ^ { ( i ) } \right) - \left( 1- y ^ { ( i ) } \right) \frac { 1} { 1- h _ { \theta } \left( x ^ { ( i ) } \right) } \frac { \partial } { \partial \theta _ { j } } h _ { \theta } \left( x ^ { ( i ) } \right) \right) 
\\ & =- \frac { 1} { m } \sum _ { i = 1} ^ { m } \left( y ^ { ( i ) } \frac { 1} { g \left( \theta ^ { T } x ^ { ( i ) } \right) } - \left( 1- y ^ { ( i ) } \right) \frac { 1} { 1- g \left( \theta ^ { T } x ^ { ( i ) } \right) } \right) \frac { \partial } { \partial \theta _ { j } } g \left( \theta ^ { T } x ^ { ( i ) } \right)
\\ & = - \frac { 1} { m } \sum _ { i = 1} ^ { m } \left( y ^ { ( i ) } \frac { 1} { g \left( \theta ^ { T } x ^ { ( i ) } \right) } - \left( 1- y ^ { ( i ) } \right) \frac { 1} { 1- g \left( \theta ^ { T } x ^ { ( i ) } \right) } \right) g \left( \theta ^ { T } x ^ { ( \text{i} ) } \right) \left( 1- g \left( \theta ^ { T } x ^ { ( i ) } \right) \right) \frac { \partial } { \partial \theta _ { j } } \theta ^ { T } x ^ { ( i ) }
\\ & = - \frac { 1} { m } \sum _ { i = 1} ^ { m } \left( y ^ { ( i ) } \left( 1- g \left( \theta ^ { T } x ^ { ( i ) } \right) \right) - \left( 1- y ^ { ( i ) } \right) g \left( \theta ^ { T } x ^ { ( i ) } \right) \right) x _ { j } ^ { ( \text{i} ) }
\\ & = - \frac { 1} { m } \sum _ { i = 1} ^ { m } \left( y ^ { ( i ) } - h _ { \theta } \left( x ^ { ( i ) } \right) \right) x _ { j } ^ { ( i ) }
\\ & = \frac { 1} { m } \sum _ { i = 1} ^ { m } \left( h _ { \theta } \left( x ^ { ( i ) } \right) - y ^ { ( i ) } \right) x _ { j } ^ { ( i ) }
\end{aligned} \right. \tag{12}
$$

上式求解过程中用到如下的公式:
$$
{ f ( x ) = \frac { 1} { 1+ e ^ { g ( x ) } } }
$$

$$
\left.\begin{aligned} 
 { \frac { \partial } { \partial x } f ( x ) = \frac { 1} { \left( 1+ e ^ { g ( x ) } \right) ^ { 2} } e ^ { g ( x ) } \frac { \partial } { \partial x } g ( x ) } 
\\ { = \frac { 1} { 1+ e ^ { g ( x ) } } \frac { e ^ { g ( x ) } } { 1+ e ^ { g ( x ) } }\frac { \partial } { \partial x } g ( x ) } 
\\ { = f ( x ) ( 1- f ( x ) ) \frac { \partial } { \partial x } g ( x ) } 
\end{aligned} 
\right. \tag{13}
$$
因此，(11)式的更新过程可以写成:
$$
\theta _ { j } : = \theta _ { j } - \alpha \frac { 1} { m } \sum _ { i = 1} ^ { m } \left( h _ { \theta } \left( x ^ { ( i ) } \right) - y ^ { ( i ) } \right) x _ { j } ^ { ( i ) } ,\quad ( j = 0\ldots n ) \tag{14}
$$
因为式中$\alpha$本来为一常量，所以一般将$\frac{1}{m}$ 省略，所以最终的$\theta$ 更新过程为:
$$
\theta _ { j } : = \theta _ { j } - \alpha \sum _ { i = 1} ^ { m } \left( h _ { \theta } \left( \text{X} ^ { ( \text{i} ) } \right) - y ^ { ( i ) } \right) x _ { j } ^ { ( \text{i} ) } ,\quad ( j = 0\ldots n ) \tag{15}
$$
另外，补充一下，3.2 节中提到求得$l(\theta)$取最大值时的$\theta$也是一样的，用梯度上升法求（9）式最大值，可得：
$$
\left.\begin{aligned} 
\theta _ { j } :& = \theta _ { j } + \alpha \frac { \partial } { \partial \theta _ { j } } \ell ( \theta )
\\ & = \theta _ { j } + \alpha \sum _ { i = 1} ^ { m } \left( y ^ { ( i ) } - h _ { \theta } \left( x ^ { ( i ) } \right) \right) x _ { j } ^ { ( \text{l} ) }  ,\quad ( j = 0\ldots n )
\end{aligned} \right. \tag{16}
$$
观察上式发现跟(14)是一样的，所以，采用梯度上升发和梯度下降法是完全一样的!





## 实践

[sklean](https://github.com/syfly007/algorithm/blob/master/MachineLearning/LogisticRegression/logistic_Sklearn.ipynb)

[tensorflow](https://github.com/syfly007/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/logistic_regression.ipynb)




# 参考资料

https://blog.csdn.net/achuo/article/details/51160101