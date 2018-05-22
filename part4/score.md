# 评价标准

预测矩阵：

|          |      实际为真       |      实际为假      |
| :------: | :-----------------: | :----------------: |
| 预测为真 | true positive (TP) | false positive (FP) |
| 预测为假 |  false negative(FN)  | true negative(TN) |



机器学习有多种评价指标，常用的有准确率（accuracy）、查准率（precision）、查全率（recall）和 F1-score。


$$
准确率（accuracy） = （TP + TN)/(TP + TN + FP + FN)
$$

​	这里$$predict$$表示预测正确的样本数，$$total$$表示总样本数。

​	准确率（accuracy）应该是最常用的评价指标，这种方式通常是可行的，但是有些特殊情况下也会遇到问题。比如有个雷达预测数据集，有99个为真，1个为假，如果模型是直接统统预测为真，准确率就为99%，看起来准确率很高，但显然不是我们想要的模型。

​	这时就需要另外2个指标：


$$
精确率(precision) = TP/(TP+FP)
$$

$$
召回率(recall) = TP/(TP+FN)
$$

​	为了便于形象理解，请看如下的文氏图。

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/640px-Precisionrecall.svg.png?1526869494881)



​	其实就是分母不同，precision的分母是全部预测为正类的样本数，recall是全部正样本数。

​	在信息检索领域，精确率和召回率又被称为**查准率**和**查全率**，

​        **查准率**＝检索出的相关信息量 / 检索出的信息总量

​	**查全率**＝检索出的相关信息量 / 系统中的相关信息总量

​	通俗的说，precison表示预测的样本中有多少是正确的，recall表示所有正样本中有多少是正确的。

​	有的时候，我们要同时衡量recall和precision，于是对它们做调和平均数(Harmonic mean)。
$$
F1 = \frac {2⋅Precison⋅Recall} {Precision + Recall}
$$
​	可以看到，recall 体现了分类模型对正样本的识别能力，recall 越高，说明模型对正样本的识别能力越强，precision 体现了模型对负样本的区分能力，precision越高，说明模型对负样本的区分能力越强。F1-score 是两者的综合。F1-score 越高，说明分类模型越稳健。

​	有的时候，我们对recall 与 precision 赋予不同的权重，表示对分类模型的偏好：
$$
F_{β}=\frac{(1+β^2)TP}{(1+β^2)TP+β^2FN+FP}=\frac{(1+β^2)⋅Precision⋅Recall}{β^2⋅Precision+Recall}
$$
​	可以看到，当 β=1，那么Fβ就退回到F1了，β其实反映了模型分类能力的偏好，β>1 的时候，precision的权重更大，为了提高Fβ，我们希望precision 越小，而recall 应该越大，说明模型更偏好于提升recall，意味着模型更看重对正样本的识别能力； 而 β<1的时候，recall 的权重更大，因此，我们希望recall越小，而precision越大，模型更偏好于提升precision，意味着模型更看重对负样本的区分能力。



参考资料：

https://www.wikiwand.com/en/F1_score

https://blog.csdn.net/matrix_space/article/details/50384518

https://www.zhihu.com/question/19645541



