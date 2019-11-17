---
title: VGG 论文阅读笔记
date: 2019-11-17 22:27:03
tags:
---
笔记阅读2015年发表于ICLR的《VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION》图像分类部分，课程实践采用的是VGG-16模型。

### 摘要
论文探讨在大规模数据集情景下，卷积网络深度对其准确率的影响。我们的主要贡献在于利用3*3小卷积核的网络结构对逐渐加深的网络进行全面的评估，结果表明加深网络深度到16至19层可极大超越前人的网络结构。这些成果是基于2014年的ImageNet挑战赛，该模型在定位和分类跟踪比赛中分别取得了第一名和第二名。同时模型在其他数据集上也有较好的泛化性。我们公开了两个表现最好的卷积神经网络模型，以促进计算机视觉领域模型的进一步研究。
#### 1.介绍
卷积网络(ConvNets)成功原因：大型公共图像数据集，如 ImageNet；高性能计算系统，如GPU或大规模分布式集群。
前人的工作：人们尝试在AlexNet的原始框架上做一些改进。比如在第一个卷积上使用较小的卷积核以及较小的滑动步长。另一种方法则是在全图以及多个尺寸上，稠密的训练和测试网络。
而本文主要关注网络的深度。为此，我们固定网络的其他参数，通过增加卷
积层来增加网络深度，这是可行的，因为我们所有层都采用小的3*3卷积核。
#### 2.卷积配置
为凸显深度对模型效果的影响，我们所有卷积采用相同配置。本章先介绍卷积网络的通用架构，再描述其评估中的具体细节，最后讨论我们的设计选择以及前人网络的比较。
##### 2.1架构
训练输入：固定尺寸224*224的RGB图像。
预处理：每个像素值减去训练集上的RGB均值。
卷积核：一系列3*3卷积核堆叠，步长为1，采用padding保持卷积后图像空间分辨率不变。
空间池化：紧随卷积“堆”的最大池化，为2*2滑动窗口，步长为2。
全连接层：特征提取完成后，接三个全连接层，前两个为4096通道，第三个为1000通道，最后是一个soft-max层，输出概率。

所有隐藏层都用非线性修正ReLu。
##### 2.2详细配置
表1中每列代表不同的网络，只有深度不同（层数计算不包含池化层）。卷积的通道数量很小，第一层仅64通道，每经过一次最大池化，通道数翻倍，知道数量达到512通道。
表2展示了每种模型的参数数量，尽管网络加深，但权重并未大幅增加，因为参数量主要集中在全连接层。
![](1.png)
##### 2.3讨论
两个3*3卷积核相当于一个5*5卷积核的感受域，三个3*3卷积核相当于一个7*7卷积核的感受域。

优点：三个卷积堆叠具有三个非线性修正层，使模型更具判别性；其次三个3*3卷积参数量更少，相当于在7*7卷积核上加入了正则化。
3. 分类框架

##### 3.1训练
训练方法基本与AlexNet一致，除了多尺度训练图像采样方法不一致。
训练采用mini-batch梯度下降法，batch size=256；
采用动量优化算法，momentum=0.9；
采用L2正则化方法，惩罚系数0.00005；dropout比率设为0.5；
初始学习率为0.001，当验证集准确率不再提高时，学习率衰减为原来的0.1倍，总共下降三次；
总迭代次数为370K（74epochs）；
数据增强采用随机裁剪，水平翻转，RGB颜色变化；
设置训练图片大小的两种方法：
定义S代表经过各向同性缩放的训练图像的最小边。
第一种方法针对单尺寸图像训练，S=256或384，输入图片从中随机裁剪
224*224大小的图片，原则上S可以取任意不小于224的值。
第二种方法是多尺度训练，每张图像单独从[Smin ,Smax ]中随机选取S来进行尺寸缩放，由于图像中目标物体尺寸不定，因此训练中采用这种方法是有效的，可看作一种尺寸抖动的训练集数据增强。
论文中提到，网络权重的初始化非常重要，由于深度网络梯度的不稳定性，不合适的初始化会阻碍网络的学习。因此我们先训练浅层网络，再用训练好的浅层网络去初始化深层网络。
##### 3.2 测试
测试阶段，对于已训练好的卷积网络和一张输入图像，采用以下方法分类：
首先，图像的最小边被各向同性的缩放到预定尺寸Q；
然后，将原先的全连接层改换成卷积层，在未裁剪的全图像上运用卷积网络，输出是一个与输入图像尺寸相关的分类得分图，输出通道数与类别数相同；
最后，对分类得分图进行空间平均化，得到固定尺寸的分类得分向量。
我们同样对测试集做数据增强，采用水平翻转，最终取原始图像和翻转图像的soft-max分类概率的平均值作为最终得分。
由于测试阶段采用全卷积网络，无需对输入图像进行裁剪，相对于多重裁剪效率会更高。但多重裁剪评估和运用全卷积的密集评估是互补的，有助于性能提升。
#### 4.分类实验
##### 4.1单尺寸评估
表3展示单一测试尺寸上的卷积网络性能
![](2.png)
##### 4.2多尺寸评估
表4展示多个测试尺寸上的卷积网络性能
![](3.png)

##### 4.3 多重裁剪与密集网络评估
表 5 展示多重裁剪与密集网络对比，并展示两者相融合的效果
![](4.png)
##### 4.4 卷积模型的融合
这部分探讨不同模型融合的性能，计算多个模型的 soft-max 分类概率的平均值来对它们的输出进行组合，由于模型的互补性，性能有所提高，这也用于比赛的最佳结果中。
表 6 展示多个卷积网络融合的结果
![](5.png)
##### 4.5 与当前最好算法的比较
表七展示对当前最好算法的对比
![](6.png)
#### 5结论
本文评估了非常深的卷积网络在大规模图像分类上的性能。结果表明深度有利于分类准确率的提升。附录中展示了模型的泛化能力，再次确认了视觉表达中深度的重要性。
```python
vgg16.py
#!/usr/bin/python
#coding:utf-8
import inspect 
import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

VGG_MEAN = [103.939, 116.779, 123.68] # 样本 RGB 的平均值

class Vgg16():
def __init__(self, vgg16_path=None):
if vgg16_path is None:
vgg16_path = os.path.join(os.getcwd(),"vgg16.npy") # os.getcwd() 方法用于返回当前工作目录。
print(vgg16_path)
self.data_dict = np.load(vgg16_path, encoding='latin1').item() # 遍历其内键值对，导入模型参数

for x in self.data_dict: #遍历 data_dict 中的每个键
print x 
def forward(self, images): 
# plt.figure("process pictures")
print("build model started")
start_time = time.time() # 获取前向传播的开始时间
rgb_scaled = images * 255.0 # 逐像素乘以 255.0（根据原论文所述的初始化步骤）
# 从 GRB 转换色彩通道到 BGR，也可使用 cv 中的 GRBtoBGR
red, green, blue = tf.split(rgb_scaled,3,3)
assert red.get_shape().as_list()[1:] == [224, 224, 1]
assert green.get_shape().as_list()[1:] == [224, 224, 1]
assert blue.get_shape().as_list()[1:] == [224, 224, 1]
# 以上 assert 都是加入断言，用来判断每个操作后的维度变化是否和预期一致 
bgr = tf.concat([
blue - VGG_MEAN[0],
green - VGG_MEAN[1],
red - VGG_MEAN[2]],3)
# 逐样本减去每个通道的像素平均值，这种操作可以移除图像的平均亮度值，该方法常用在灰度图像上
assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

# 接下来构建 VGG 的 16 层网络（包含 5 段卷积，3 层全连接），并逐层根据命名空间读取网络参数
# 第一段卷积，含有两个卷积层，后面接最大池化层，用来缩小图片尺寸
self.conv1_1 = self.conv_layer(bgr, "conv1_1")
# 传入命名空间的 name，来获取该层的卷积核和偏置，并做卷积运算，最后返回经过经过激活函数后的值
self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
# 根据传入的 pooling 名字对该层做相应的池化操作
self.pool1 = self.max_pool_2x2(self.conv1_2, "pool1")

# 下面的前向传播过程与第一段同理
# 第二段卷积，同样包含两个卷积层，一个最大池化层
self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
self.pool2 = self.max_pool_2x2(self.conv2_2, "pool2")

# 第三段卷积，包含三个卷积层，一个最大池化层
self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
self.pool3 = self.max_pool_2x2(self.conv3_3, "pool3")

# 第四段卷积，包含三个卷积层，一个最大池化层
self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2") 
self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
self.pool4 = self.max_pool_2x2(self.conv4_3, "pool4")

# 第五段卷积，包含三个卷积层，一个最大池化层
self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
self.pool5 = self.max_pool_2x2(self.conv5_3, "pool5")

# 第六层全连接
self.fc6 = self.fc_layer(self.pool5, "fc6") # 根据命名空间 name 做加权求和运算
assert self.fc6.get_shape().as_list()[1:] == [4096] # 4096 是该层输出后的长度
self.relu6 = tf.nn.relu(self.fc6) # 经过 relu 激活函数

# 第七层全连接，和上一层同理
self.fc7 = self.fc_layer(self.relu6, "fc7")
self.relu7 = tf.nn.relu(self.fc7)

# 第八层全连接
self.fc8 = self.fc_layer(self.relu7, "fc8")
# 经过最后一层的全连接后，再做 softmax 分类，得到属于各类别的概率
self.prob = tf.nn.softmax(self.fc8, name="prob")

end_time = time.time() # 得到前向传播的结束时间
print(("time consuming: %f" % (end_time-start_time)))

self.data_dict = None # 清空本次读取到的模型参数字典

# 定义卷积运算 
def conv_layer(self, x, name): 
with tf.variable_scope(name): # 根据命名空间找到对应卷积层的网络参数 
w = self.get_conv_filter(name) # 读到该层的卷积核
conv = tf.nn.conv2d(x, w, [1, 1, 1, 1], padding='SAME') # 卷积计算
conv_biases = self.get_bias(name) # 读到偏置项
result = tf.nn.relu(tf.nn.bias_add(conv, conv_biases)) # 加上偏置，并做激活计算
return result 

# 定义获取卷积核的函数 
def get_conv_filter(self, name): 
# 根据命名空间 name 从参数字典中取到对应的卷积核
return tf.constant(self.data_dict[name][0], name="filter") 
# 定义获取偏置项的函数 
def get_bias(self, name): 
# 根据命名空间 name 从参数字典中取到对应的卷积核
return tf.constant(self.data_dict[name][1], name="biases") 
# 定义最大池化操作 
def max_pool_2x2(self, x, name):
return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name) 
# 定义全连接层的前向传播计算 
def fc_layer(self, x, name): 
with tf.variable_scope(name): # 根据命名空间 name 做全连接层的计算 
shape = x.get_shape().as_list() # 获取该层的维度信息列表 
print "fc_layer shape ",shape dim = 1
for i in shape[1:]: 
dim *= i # 将每层的维度相乘 
# 改变特征图的形状，也就是将得到的多维特征做拉伸操作，只在进入第六层全连接层做该操作
x = tf.reshape(x, [-1, dim]) 
w = self.get_fc_weight(name)# 读到权重值
b = self.get_bias(name) # 读到偏置项值 

result = tf.nn.bias_add(tf.matmul(x, w), b) # 对该层输入做加权求和，再加上偏置
return result 

# 定义获取权重的函数 
def get_fc_weight(self, name): # 根据命名空间 name 从参数字典中取到对应的权重 
return tf.constant(self.data_dict[name][0], name="weights") 
```
