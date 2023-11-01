# 基于PyTorch的LeNet-5实现
## 一、神经网络基本原理
卷积神经网络主要由卷积层、池化层、全连接层构成。卷积层通过对输入图像进行卷积操作来提取图像特征。池化层对输入的特征图片进行压缩，简化网络计算复杂度。全连接层连接所有的特征，并将输出值送给分类器。
## 二、LeNet-5基本结构
### 1.输入层
用以接收输入的图像数据。在CIFAR10数据集上，输入为32x32x3的图像数据。图像长宽为32，以及RGB三个颜色通道。
### 2.卷积层1
卷积层1包括6个卷积核，每个卷积核大小为5x5，步长为1，填充为0，输出6个通道的28x28的特征图
### 3.采样层1
采样层1采用最大池化操作，窗口大小为2x2，步长为2.每个池化操作从2x2的窗口中选择最大值，输出6个通道的14x14的特征图。采样层可减少特征图大小，并对于轻微的位置变化保持不变性。
### 4.卷积层2
卷积层2包括16个卷积核，每个卷积核大小为5x5，步长为1，填充为0.输出16个通道的10x10的特征图
### 5.采样层2
操作同采样层1，输出16个通道的5x5的特征图
### 6.全连接层1
全连接层1将每个通道的特征图拉伸为一维向量（16x5x5），并通过带有120个神经元的全连接层连接。120为设计者根据实验得到的最佳值。
### 7.全连接层2
全连接层2将120个神经元连接到84个神经元。84为设计者根据实验得到的最佳值。
### 8.输出层
输出层由10个神经元组成，对应10组图片分类，并输出最终的分类结果。

## 三、PyTorch代码实现
LeNet类初始化：将各层作为成员变量存储。
```python
    def __init__(self,
    channel1=CHANNEL_1, # 卷积层1的卷积核数量。默认为6
    channel2=CHANNEL_2, # 卷积层2的卷积核数量。默认为16
    fc_count=FC_COUNT): # 全连接层数量，默认为3
        super(LeNet5, self).__init__()  # 父类构造方法
        self.conv1 = nn.Conv2d(3, channel1, 5,1,0) # 卷积层1
        self.pool = nn.MaxPool2d(2, 2)  #
        self.conv2 = nn.Conv2d(channel1, channel2, 5, 1, 0)
        self.fc1=nn.Linear(channel2 * 25, 120)
        self.fc2=nn.Linear(120, 84)
        self.fc=[]
        for i in range(fc_count-3):
            self.fc.append(nn.Linear(84,84))
        self.fc3=nn.Linear(84, 10)
        self.channel2=channel2
        self.dataset=Dataset()
        print(f'Init completed with channel1={channel1},channel2={channel2},fc_count={len(self.fc)+3}')
```
## 四、实验结果截图
 <!-- 参考：
 https://blog.csdn.net/yjl9122/article/details/70198357
% https://zhuanlan.zhihu.com/p/616996325 -->