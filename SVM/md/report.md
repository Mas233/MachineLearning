# SVM小实验

211250143 王铭嵩

## 一、SVM算法原理

对于支持向量机，一个点（样例）对应的Margin是其到分界超平面的垂直距离。具有最小Margin的点称为支持向量。SVM的核心要义是最大化所有训练样本的最小间隔Margin。

## 二、代码实现

实验使用sklearn的svm库来实现支持向量机。

- 数据获取：
  
  依据作业要求实现
  
  ![](C:\Users\Lenovo\AppData\Roaming\marktext\images\2023-11-29-20-37-07-image.png)

- 创建分类器并处理数据：
  
  ![](C:\Users\Lenovo\AppData\Roaming\marktext\images\2023-11-29-20-37-49-image.png)

- 获取支持向量并输出：
  
  ![](C:\Users\Lenovo\AppData\Roaming\marktext\images\2023-11-29-21-11-07-image.png)

- 获取超平面参数并输出：
  
  ![](C:\Users\Lenovo\AppData\Roaming\marktext\images\2023-11-29-21-11-28-image.png)

- 利用`matplotlib.pyplot`包来可视化结果。其中支持向量被突出显示；超平面和间隔以直线方式打印：
  
  ![](C:\Users\Lenovo\AppData\Roaming\marktext\images\2023-11-29-21-13-17-image.png)

## 三、实验结果

支持向量及超平面参数输出：

![](C:\Users\Lenovo\AppData\Roaming\marktext\images\2023-11-29-21-10-15-image.png)

可视化：

![](D:\Code\MachineLearning\SVM\python\result.png)

支持向量为被突出表示的点；灰线为超平面。
