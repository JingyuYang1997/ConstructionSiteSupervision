###   基于Mask RCNN的工地监测系统

本项目基于Mask RCNN算法和MOCS数据集，设计了一种工地监测系统。

#### 使用说明

##### 训练

1. 下载MOCS数据集 ([下载地址](http://www.anlab340.com/Archives/IndexArctype/index/t_id/17.html))，解压至MOCS文件夹

2. 下载imagenet预训练模型 ([下载地址](https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5))，放入pretrained_models文件夹

3. ```shell
   cd ./mrcnn/samples/mocs 
   python mocs.py train --model=imagenet
   ```

本项目运行环境：

```shell
15 vCPU Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz
RTX 3090(24GB) * 1
Python  3.8(ubuntu18.04) 
TensorFlow  1.15.5 
Cuda  11.4
```

训练一次用时大概60小时

##### Demo

1. 将想要测试的图片放入./mrcnn/samples/mocs /images文件夹中
2. 打开jupyter notebook，运行demo_mocs.ipynb, 或者运行run_demo.py。

#### 部分结果展示



<img src="mrcnn\samples\mocs\results\0018730.jpg" alt="0018730" style="zoom: 70%;" />

<img src="mrcnn\samples\mocs\results\0000442.jpg" alt="0000442" style="zoom: 70%;" />

<img src="mrcnn\samples\mocs\results\0000243.jpg" alt="0000243" style="zoom: 70%;" />

<img src="mrcnn\samples\mocs\results\0000291.jpg" alt="0000291" style="zoom: 70%;" />

<img src="mrcnn\samples\mocs\results\0000424.jpg" alt="0000424" style="zoom: 70%;" />

<img src="mrcnn\samples\mocs\results\0000537.jpg" alt="0000537" style="zoom: 70%;" />

<img src="mrcnn\samples\mocs\results\0008485.jpg" alt="0008485" style="zoom: 70%;" />

<img src="mrcnn\samples\mocs\results\0000674.jpg" alt="0000674" style="zoom: 70%;" />

<img src="mrcnn\samples\mocs\results\0000394.jpg" alt="0000394" style="zoom: 70%;" />

<img src="mrcnn\samples\mocs\results\0000284.jpg" alt="0000284" style="zoom: 70%;" />















