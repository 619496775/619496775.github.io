---
title: 初学tensorflow
date: 2019-08-26 23:49:07
tags:
---

{% asset_img tf_logo.png %}
<!-- more -->
###  一、深度学习框架的选择

下面再简单介绍一下其他深度学习框架的特点：

（1） Caffe：卷积神经网络框架，专注于卷积神经网络和图像处理，因为是基于C++语言，所以执行速度非常的快。

（2） PyTorch：动态computation graph！！！（笔者学习Tensorflow一段后，便会转学PyTorch试试看）

（3） Theano：因其定义复杂模型很容易，在研究中比较流行。

（4） CNTK：微软开发的，微软称其在语音和图像识别方面比其他框架更有优势。不过代码只支持C++.

Tensorflow的一些特性就不再说了，网络上相关资料也有很多。

下面就介绍一下Tensorflow的安装，笔者的安装顺序是首先安装Anaconda、然后安装Tensorflow、再安装Pycharm。

### 二、安装Anaconda
conda activate
conda deactivate
### 三、建立、激活、安装Tensorflow

| 用法 | 简介 |
| --- | --- |
| conda info | conda 基本信息，包括所在平台，版本，路径等 |
| conda list [-n envName] | 安装了的软件包 |
| conda search packageName | 搜索软件包 |
| conda create envName | 创建一个环境 |
| conda install |  安装软件包|
| conda update |更新软件包  |
| conda remove | 删除软件包 |

### 四、PyCharm IDE
安装主要依赖的Python类库

| 软件包名称 | 简介 |
| --- | --- |
| Numpy | Python开源的数值计算扩展。提供了如矩阵数据类型、失量处理，以及精密的运算库 |
|SciPy  |  再Numpy的基础上增加了众多科学计算的常用库。如线性代数、常微分方程数值求解、信号处理、图像处理、稀疏矩阵等|
| pandas | 解决数据分析任务 |
| Matpoltlib |Python 2D绘图领域  |
| Seaborn | Matpoltlib封装，使得绘制更加简单 |
| Scikit-Learn |分类、回归、聚类、数据降维、模型选择和数据预处理  |
| XGBoost | GBDT的一种实现类库 |
| OpenCV | 视觉处理 |
### 五、总结
