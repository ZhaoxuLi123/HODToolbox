# HODToolbox

[English](README.md) | 简体中文

**HODToolbox是高光谱对象级目标检测任务（hyperspectral object detection，HOD）的开源工具箱，也是[SpecDETR](https://www.sciencedirect.com/science/article/abs/pii/S092427162500186)的配套代码。
HODToolbox实现了从传统像素级高光谱目标检测任务（hyperspectral target detection，HTD）任务到HOD任务的范式转换，集成以下核心功能：**

- **将传统HTD数据集转换为HOD数据集，由单条先验光谱即可生成大规模训练图像集；**
   
- **主流视觉目标检测网络在HOD数据集上训练和测试；**

- **传统HTD方法的检测分数图结果转化对象级预测框结果；**

- **目标检测网络和HTD方法结果定量评估和可视化。**




<br>


## 安装

1. 安装SpecDETR项目，安装细节查看
[https://github.com/ZhaoxuLi123/SpecDETR](https://github.com/ZhaoxuLi123/SpecDETR/blob/main/README.md) 


2. 克隆该仓库至本地

    ```bash
    HODToolbox_ROOT=/path/to/clone/HODToolbox
    git clone https://github.com/ZhaoxuLi123/HODToolbox $HODToolbox_ROOT
    ```
   
3. 安装依赖包
    ```bash
    cd $HODToolbox_ROOT
    pip install -r requirements.txt
    ```

<br>

## 数据集

我们构建了首个高光谱多类别点目标/亚像元/微小目标对象级检测数据集SPOD，并且将Avon、SanDiego和MUUFLGulfport三个公共的HTD数据集转为HOD数据集。

- **SPOD数据集：**
  - **30波段数据（基准评测所用数据）: [Baidu Drive](https://pan.baidu.com/s/1fySVhp4w2coz1vwvB6aSgw?pwd=2789) (key: 2789) 或者 [OneDrive](https://dh7n-my.sharepoint.com/:u:/g/personal/vv_s3_tm9_site/Ea8D-QY1zoxKq8a1xCj0XXoB4dNWd-M2BM3FvYV042JHXw)**
  - **150波段数据（原始数据）: [Baidu Drive](https://pan.baidu.com/s/1WXuWb694J4QGJaTQ9Qxg_Q?pwd=2789) (key: 2789) 或者 [OneDrive](https://dh7n-my.sharepoint.com/:u:/g/personal/vv_s3_tm9_site/EQ2YXMp0xcRPrA3hFoviSrgB7jTrXBg4RtAPr1w8cIoHBw)**

- **Avon数据集: [Baidu Drive](https://pan.baidu.com/s/13yIPxUulRAa0-s_O_eFL8w?pwd=2789) (key: 2789) 或者 [OneDrive](https://dh7n-my.sharepoint.com/:u:/g/personal/vv_s3_tm9_site/EXI7DG5slNdCkAla1QqNUagBA2V_cXgw_Oj8p5tHijttAg)**

- **SanDiego数据集: [Baidu Drive](https://pan.baidu.com/s/1bKUFdZC0GQYDUSPRh5QBpw?pwd=2789) (key: 2789) 或者 [OneDrive](https://dh7n-my.sharepoint.com/:u:/g/personal/vv_s3_tm9_site/EaZqpRVz_nxPgG8ufedM0U0BslexIeE138_RGYXOcMgjpw)**

- **MUUFLGulfport数据集: [Baidu Drive](https://pan.baidu.com/s/1xWA45V92eGEs29tJvNl8AA?pwd=2789) (key: 2789) 或者 [OneDrive](https://dh7n-my.sharepoint.com/:u:/g/personal/vv_s3_tm9_site/EfsT0_HjSQJLj5c-Hy777MUBS9a0wBTYQlktLtm5rz4E5w)**

将四个数据集放置到`./datasets/`文件夹下。`./datasets/`文件夹结构如下：
  ```
  ├──./datasets/
  │    ├── SPOD_150b_8c
  │    │    ├── annotations
  │    │    │    ├── all.json
  │    │    │    ├── train_split.json
  │    │    │    ├── test_split.json
  │    │    ├── data
  │    │    │    ├── 100001.npy
  │    │    │    ├── ...
  │    │    ├── mask
  │    │    │    ├── 100001mask.mat
  │    │    │    ├── ...
  │    │    ├── color
  │    │    │    ├── 100001.png
  │    │    │    ├── ... 
  │    │    ├── target.mat 
  │    ├── SPOD_30b_8c
  │    │    ├── annotations
  │    │    │    ├── train.json
  │    │    │    ├── test.json
  │    │    ├── train
  │    │    │    ├── 100007.npy
  │    │    │    ├── ...
  │    │    ├── test
  │    │    │    ├── 200001.npy
  │    │    │    ├── ...  
  │    │    ├── color
  │    │    │    ├── 100007.png
  │    │    │    ├── ... 
  │    │    ├── mask_gt
  │    │    │    ├── 200001.mat
  │    │    ├── target.mat  
  │    ├── Avon
  │    ├── ...
  ```
以下是数据集格式说明：

- `./data`, `./train`, `./test` 存储图像数据，采用npy文件格式，形状为`高*宽*波段数`。

- `./annotations`, 存储COCO格式的真值标签, `train.json`为训练集标签，`test.json`为测试集标签。

- `./mask`, 存储仿真目标的像素信息，采用mat文件格式，每个mat文件包含`abundance_masks`、`cat_mask`、`target_mask`，`abundance_masks`形状为`高*宽*4`,第一层表示各像素上目标丰度，第2-4层表示目标像素各类目标光谱比例（SPOD数据集一个目标最多由3种光谱组成）,`target_mask`形状为`高*宽`，0表示是背景像素，1，2，3，...，n表示不同目标实例的像素，`cat_mask`形状为`高*宽`，0表示是背景像素，1，2，3，...，n表示不同目标类型的像素。

- `./color`, 存储高光谱数据的彩色图，采用png文件格式。

- `./mask_gt`, 存储测试图像mask真值标签，用于HTD方法的评估，采用mat文件格式，形状为`目标类别数*高*宽`。

- `target.mat`, 目标先验光谱库，用于HTD方法，形状为`目标类别数*光谱数*波段数`。

### SPOD数据集

SPOD数据集是**首个高光谱多类点目标/亚像元目标对象级检测数据集**，包含8类目标，大部分目标像素为混合像元，C1类和C1类目标均为极低丰度的单像素目标，C4和C5为易混淆目标，C6、C7和C8为多类光谱组成的目标。

在json标签文件中，C1、C2、C3、C4、C5、C6、C7和C8类别分别用CB、MP、FG、NP、LO、K_N、P_O和V_Y_W表示。

SPOD数据集的高光谱数据包含150光谱通道，为增加评估难度，我们将光谱通道下采样到30。

SPOD_150b_8c为150波段的数据，SPOD_30b_8c是我们在实验中所采用的数据集。

[SPOD_dataset_covert.py](SPOD_dataset_covert.py) 提供了SPOD_150b_8c到SPOD_30b_8c转换代码。





### HTD数据集

传统HTD数据集通常只有单张测试图像，提供几条甚至1条目标光谱做先验信息。而HOT任务需要带有对象级标签的大规模训练图像集身上进行。
我们将SPOD数据集的仿真方法进行扩展，可由单条先验目标光谱生成HOD任务的训练集。

[HTDDataset2HODDataset.py](HTDDataset2HODDataset.py) 提供了传统HTD数据集到HOD数据集的转化功能。

<br>

## 基准测试

我们在SPOD数据集上对当前主流的视觉目标检测网络和HTD方法进行了测试。

视觉目标检测网络测试在MMDetection框架下进行，我们在[SpecDETR](https://github.com/ZhaoxuLi123/SpecDETR)项目提供了对应的config配置文件，实验可复现。

HTD方法测试主要依托团队前期构建的HTD工具箱实现，计划在近期开源。

在SPOD、Avon、SanDiego和MUUFLGulfport数据集的测试结果可从以下链接下载：
[Baidu Drive](https://pan.baidu.com/s/1lVtY5mPzhcovB89t_EjtnA?pwd=2789) (key:2789) 或者 [OneDrive](https://dh7n-my.sharepoint.com/:u:/g/personal/vv_s3_tm9_site/EeqwBubB5o1KvC4Zu9kCLDEBRhUGPN1Ux25ZonXlk20GVQ)
 
下载的zip文件解压后放置于`./eval_result/`文件夹下，该文件夹结构如下:
  ```
  ├──./eval_result/
  │    ├── SPODResult
  │    │    ├── HOD
  │    │    │    ├── cocojson
  │    │    │    │    ├── atss_r50_fpn_dyhead.json
  │    │    │    │    ├── ...
  │    │    │    ├── result
  │    │    │    │    ├── atss_r50_fpn_dyhead  
  │    │    │    │    │    ├─result.pkl
  │    │    │    ├── ...
  │    │    ├── HTD
  │    │    │    ├── cocojson
  │    │    │    │    ├── ASD.json
  │    │    │    │    ├── ...
  │    │    ├── eval
  │    │    │    ├── 0HODapar.txt
  │    │    │    ├── ... 
  │    │    ├── img
  │    │    │    ├── 200389_atss_r50_fpn_dyhead.png
  │    │    │    ├── ...    
  │    ├── AvonResult
  │    ├── ...
  ```
以下是结构的具体说明：

- `./XXXResult/HOD/result`存储MMDetection输出的目标检测网络测试结果文件，文件格式为pkl。
- `./XXXResult/HOD/cocojson`存储从目标检测网络pkl结果文件转换的COCO格式测试结果文件，文件格式为json。
- `./XXXResult/HTD/result`存储HTD方法输出的检测分数图，文件格式为mat，每个mat文件包含单张图像的测试结果，形状为`目标类别数*高*宽`。 
- `./XXXResult/HTD/cocojson`存储从目标检测网络pkl结果文件转换的COCO格式测试结果文件，文件格式为json。
- `./XXXResult/eval`存储定量评估结果，文件格式为txt。
- `./XXXResult/img`存储可视化结果，文件格式为png、eps或者pdf。


在SPOD数据集上视觉目标检测网络部分测试结果如下：

|                                 方法                                 |   骨干网络       |  输入图像尺寸  |       mAP50:95        |        mAP25        |        mAP50        |          mAP75        |      FLOPs       |        Params    |
|:------------------------------------------------------------------:|:------------:|:------:|:---------------------:|:-------------------:|:-------------------:|:---------------------:|:---------------------:|:---------------:|
|                          **Faster R-CNN**                          | **ResNet50** |   x4   |     0.197    |   0.377    |    0.374    |     0.179     |     68.8G     | 41.5M   |
|                          **Faster R-CNN**                          | **RegNetX** |   x4   |     0.227    |   0.379   |    0.378    |    0.242    |     57.7G     | 31.6M   |
|                          **Faster R-CNN**                          | **ResNeSt50** |   x4   |     0.246    |   0.316    |    0.316    |     0.277     |     185.1G     | 44.6M   |
|                          **Faster R-CNN**                          | **ResNeXt101** |   x4   |     0.220    |   0.368   |    0.366   |     0.231     |     128.4G     | 99.4M   |
|                          **Faster R-CNN**                          | **HRNet** |   x4   |     0.320    |   0.404    |    0.402   |     0.345     |     104.4G     | 63.2M   |
|                          **TOOD**                          | **ResNeXt101** |   x4   |     0.304    |   0.464    |    0.440    |     0.303    |     114.3G     | 97.7M   |
|                          **CentripetalNet**             | **HourglassNet104** |   x4   |    0.695    |   0.829    |   0.805    |     0.673     |     501.3G     | 205.9M   |
|                          **CornerNet**                          | **HourglassNet104** |   x4   |     0.626    |   0.736   |    0.712    |     0.609     |    462.6G     | 201.1M   |
|                          **RepPoints**                          | **ResNet50** |   x4   |     0.207   |   0.691   |    0.572   |     0.074     |     54.1G     | 36.9M   | |
|                          **RepPoints**                          | **ResNeXt101** |   x4   |   0.485    |   0.806    |    0.790   |     0.540     |     75.0G     | 58.1M   |
|                          **RetinaNet**                          | **EfficientNet** |   x4   |    0.462   |  0.836    |    0.811   |     0.466     |     36.1G     | 18.5M   |
|                          **RetinaNet**                          | **PVTv2-B3** |   x4   |   0.426    |  0.757   |    0.734    |    0.442     |     71.3G     | 52.4M   |
|                          **DeformableDETR**                          | **ResNet50** |   x4   |     0.231    |   0.692    |    0.560    |     0.147     |     58.7G     | 41.2M   |
|                          **DINO**                          | **ResNet50** |   x4   |    0.168   |   0.491   |    0.418   |    0.097    |    86.3G     | 47.6M   |
|                          **DINO**                          | **Swin-L** |   x4   |    0.757    |   0.852   |    0.842    |    0.764     |     203.9G     | 218.3M   |
|                          **SpecDETR**                          | -- |   x1  |     **0.856**    |  **0.938**    |    **0.930**   |     **0.863**   |     139.7G     | 16.1M   |



在SPOD数据集上HTD方法的部分测试结果如下：

|  方法    |     mAP50:95     |   mAP25   |        mAP50        |          mAP75        |
|:----------:|:----------------:|:---------:|:-------------------:|:---------------------:|
| **ASD**	 |     0.182 	      |  0.286 	  | 0.260 	 | 0.182  |
| **CEM** |     0.040 	      |  0.122 	  | 0.075 	 | 0.035 |
| **CRBBH**	 |    0.036 	 	     |  0.129 	  | 0.083 	 | 0.028 |
| **CSRBBH**	 |   0.034 	    	   |  0.116 	  | 0.076 	 | 0.028 |
| **HSS**	 |     0.073 	      |  0.303 	  | 0.179 	 | 0.058 |
| **IRN**	 | 0.000 	       	  |  0.002 	  | 0.001 	 | 0.000 |
| **KMSD**	 |   0.108 	    	   |  0.285 	  | 0.207 	 | 0.095 |
|  **KOSP**	 |     0.017 	      |  0.083 	  | 0.044 	 | 0.014 |
|  **KSMF**	 | 0.003 	       	  |  0.015 	  | 0.009 	 | 0.002 |
|  **KTCIMF**	 | 0.001 	       	  |  0.008 	  | 0.002 	 | 0.000 |
|  **LSSA**	 |  0.041 	      	  |  0.093 	  | 0.071 	 | 0.037 |
|  **MSD**	 | 0.248 	       	  |  0.521 	  | 0.402 	 | 0.228 |
|  **OSP**	 | 0.031 	       	  |  0.108 	  | 0.063 	 | 0.027 |
|  **SMF**	 |  0.003 	     	   |  0.016 	  | 0.008 	 | 0.002 |
|  **SRBBH**	 | 0.019 	       	  |  0.092 	  | 0.051 	 | 0.013 | 
|  **SRBBH_PBD**	 | 0.013 	       	  |  0.088 	  | 0.038 	 | 0.007 |
|  **TCIMF**	 | 0.009 	       	  |  0.061 	  | 0.025 	 | 0.007 |
|  **TSTTD**	 | 0.044 	        	 |  0.057 	  | 0.055 	 | 0.043 |
|  **SpecDETR**  	 |    **0.856**     | **0.938** 	 | **0.930** 	 | **0.863** |

更多评测结果查看 `./eval_result/SPODResult/eval`

<br>



## HOD任务训练和推理


[SpecDETR](https://github.com/ZhaoxuLi123/SpecDETR)项目依托MMdetection项目构建，支持主流目标检测网络在SPOD数据集上训练和推理。

`SpecDETR/configs/VisualObjectDetectionNetwork`文件夹下提供了实验中所用的配置文件。

[SpecDETR](https://github.com/ZhaoxuLi123/SpecDETR)项目也支持由[HTDDataset2HODDataset.py](HTDDataset2HODDataset.py)生成的HOD数据集的训练和推理。需要进行以下修改：

- `SpecDETR/mmdet/datasets/hsi/HSI.py`添加HOD数据集目标类别：

   ```python
    class HSIDataset(CocoDataset):
        METAINFO = {
                'classes':
                ('CB',  'add new class',  'add new class'),
                'palette':
                [(220, 20, 60), (add new color),(add new color)]
                }
  ```
- `SpecDETR/configs/VisualObjectDetectionNetwork/_base_/datasets/hsi_detection4x.py` 修改数据集配置文件：
  
  ```python
    data_root = 'new_HOD_dataset_path'   # ← Update to the local path of the new dataset
    train_pipeline = [
        dict(type='LoadHyperspectralImageFromFiles', normalized_basis=5000),  # Normalized_basis is the normalization constant in the paper. Update to match the new dataset. 
                        ]
  ```
- 修改模型配置文件,需要修改网络输入的通道数以及目标类别数，以`SpecDETR/configs/VisualObjectDetectionNetwork/dino-5scale_swin-l-100e_hsi4x.py`为例：
    ```python
    model = dict(
        # Backbone configuration
        backbone=dict(
            in_channels=30,  # ← Update to match your dataset's spectral dimension (e.g., 30 for SPOD_30b_8c)
        ),
        
        # Detection head configuration
        bbox_head=dict(
            num_classes=8,  # ← Update to match your dataset's category count (e.g., 8 for SPOD)
        )
    )
    ```

运行`SpecDETR/train.py`实现模型训练。

运行`SpecDETR/test.py`实现模型推理。


<br>

## HTD结果文件转换

[HTDResult2Json.py](HTDResult2Json.py)实现从传统HTD方法的检测分数图到对象集目标检测的预测矩形框结果的转换。

<br>

## MMDetection结果文件转换

[MMDdetResult2Json.py](MMDdetResult2Json.py)实现从MMDetection项目输出的pkl结果文件到HODToolbox评估所用的json文件转换。

<br>

## 结果定量评估

[ResultEval.py](ResultEval.py)可对转化后的json结果文件评估COCO格式的Average Precision（AP）和Average Recall（AR）指标。
此外，[ResultEval.py](ResultEval.py)也提供一些小目标检测任务中所用的Precision\Recall\F1分数\虚警率指标的评估结果。

<br>

## 结果可视化

运行[ResultShow.py](ResultShow.py)得到可视化结果。

<br>

## 引用
```
@article{li2025specdetr,
  title={SpecDETR: A transformer-based hyperspectral point object detection network},
  author={Li, Zhaoxu and An, Wei and Guo, Gaowei and Wang, Longguang and Wang, Yingqian and Lin, Zaiping},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={226},
  pages={221--246},
  year={2025},
  publisher={Elsevier}
}
```


<br>

## 联系
**如有任何疑问，欢迎与我联系[lizhaoxu@nudt.edu.cn](lizhaoxu@nudt.edu.cn)。**