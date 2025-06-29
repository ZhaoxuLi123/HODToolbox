# HODToolbox  

English | [简体中文](README_cn.md)  

**HODToolbox is an open-source toolbox for hyperspectral object detection (HOD) tasks and serves as the companion code for [SpecDETR](https://github.com/ZhaoxuLi123/SpecDETR).** 

**HODToolbox facilitates a paradigm shift from the traditional pixel-level hyperspectral target detection (HTD) task to the HOD task, integrating the following core functionalities:**  

- **Convert HTD datasets into HOD datasets, allowing the generation of large-scale training image sets from a single prior target spectrum.**  

- **Train and test mainstream visual object detection networks and SpecDETR on HOD datasets.**  

- **Transform detection score maps from HTD methods into object-level prediction bounding boxes.**  

- **Quantitative evaluation and visualization of results from object detection networks and HTD methods.**

<br>


## Installation  

1. Install the SpecDETR project. For installation details, refer to:  
[https://github.com/ZhaoxuLi123/SpecDETR](https://github.com/ZhaoxuLi123/SpecDETR/blob/main/README.md)  

2. Clone this repository locally:  
    ```bash
    HODToolbox_ROOT=/path/to/clone/HODToolbox  
    git clone https://github.com/ZhaoxuLi123/HODToolbox $HODToolbox_ROOT  
    ```  

3. Install the required dependencies:  
    ```bash
    cd $HODToolbox_ROOT  
    pip install -r requirements.txt  
    ```


<br>
## Datasets

We have constructed the first hyperspectral multi-class point/subpixel/tiny object detection benchmark dataset **SPOD**, and converted three public HTD datasets (**Avon**, **SanDiego**, and **MUUFLGulfport**) into HOD datasets.  

- **SPOD Dataset:**  
  - 30-band data (benchmark evaluation data): [Baidu Drive](https://pan.baidu.com/s/1fySVhp4w2coz1vwvB6aSgw?pwd=2789) (key: 2789) or [OneDrive](https://dh7n-my.sharepoint.com/:u:/g/personal/vv_s3_tm9_site/Ea8D-QY1zoxKq8a1xCj0XXoB4dNWd-M2BM3FvYV042JHXw)  
  - 150-band data (raw data): [Baidu Drive](https://pan.baidu.com/s/1WXuWb694J4QGJaTQ9Qxg_Q?pwd=2789) (key: 2789) or [OneDrive](https://dh7n-my.sharepoint.com/:u:/g/personal/vv_s3_tm9_site/EQ2YXMp0xcRPrA3hFoviSrgB7jTrXBg4RtAPr1w8cIoHBw)  

- **Avon Dataset**: [Baidu Drive](https://pan.baidu.com/s/13yIPxUulRAa0-s_O_eFL8w?pwd=2789) (key: 2789) or [OneDrive](https://dh7n-my.sharepoint.com/:u:/g/personal/vv_s3_tm9_site/EXI7DG5slNdCkAla1QqNUagBA2V_cXgw_Oj8p5tHijttAg)  

- **SanDiego Dataset**: [Baidu Drive](https://pan.baidu.com/s/1bKUFdZC0GQYDUSPRh5QBpw?pwd=2789) (key: 2789) or [OneDrive](https://dh7n-my.sharepoint.com/:u:/g/personal/vv_s3_tm9_site/EaZqpRVz_nxPgG8ufedM0U0BslexIeE138_RGYXOcMgjpw)  

- **MUUFLGulfport Dataset**: [Baidu Drive](https://pan.baidu.com/s/1xWA45V92eGEs29tJvNl8AA?pwd=2789) (key: 2789) or [OneDrive](https://dh7n-my.sharepoint.com/:u:/g/personal/vv_s3_tm9_site/EfsT0_HjSQJLj5c-Hy777MUBS9a0wBTYQlktLtm5rz4E5w)

Place the 4 datasets to the folder `./datasets/`.This folder  has the following structure:
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

Here is the dataset format description:

- `./data`, `./train`, `./test` store image data in npy file format, with shape `height*width*number of bands`.

- `./annotations` stores ground truth labels in COCO format, `train.json` contains training set labels, `test.json` contains test set labels.

- `./mask` stores pixel-level information of simulated targets in mat file format. Each mat file contains `abundance_masks`, `cat_mask`, and `target_mask`. `abundance_masks` has shape `height*width*4`: the first layer represents target abundance at each pixel, layers 2-4 represent spectral proportions of different target types at target pixels (SPOD dataset allows a target to consist of up to 3 spectral signatures). `target_mask` has shape `height*width`: 0 represents background pixels, 1, 2, 3, ..., n represent pixels of different target instances. `cat_mask` has shape `height*width`: 0 represents background pixels, 1, 2, 3, ..., n represent pixels of different target categories.

- `./color` stores color images of hyperspectral data in png file format.

- `./mask_gt` stores ground truth mask labels for test images, used for HTD method evaluation, in mat file format with shape `number of target categories*height*width`.

- `target.mat` is the target spectral library for HTD methods, with shape `number of target categories*number of spectra*number of bands`.


### SPOD Dataset  

The SPOD dataset is the **first hyperspectral multi-class point /sub-pixel object detection dataset**, containing 8 object categories. Most object pixels are mixed pixels whose object abundances are lower than 1, with C1 and C2 classes being extremely low-object-abundance single-pixel objects, C4 and C5 being easily confusable objects, and C6, C7, and C8 being objects composed of multiple spectral signatures.  

In the JSON annotation files, the categories C1, C2, C3, C4, C5, C6, C7, and C8 are represented by CB, MP, FG, NP, LO, K_N, P_O, and V_Y_W respectively.

The hyperspectral data in the SPOD dataset consists of 150 spectral channels. To increase evaluation difficulty, we downsampled the spectral channels to 30.  

- **SPOD_150b_8c**: The original 150-band dataset.  
- **SPOD_30b_8c**: The dataset used in our experiments (30 bands).  

The conversion script [SPOD_dataset_covert.py](SPOD_dataset_covert.py) provides the code to transform **SPOD_150b_8c** into **SPOD_30b_8c**.  



### HTD Datasets  

Conventional HTD datasets typically contain only a single test image with limited prior information (often just one or a few target spectra). However, HOD tasks require large-scale training image sets with object-level annotations.  

We have extended the simulation approach from the SPOD dataset to enable generating training sets for HOD tasks using just a single prior target spectrum.  

The conversion script [HTDDataset2HODDataset.py](HTDDataset2HODDataset.py) provides functionality to transform traditional HTD datasets into COCO-style HOD datasets.  



<br>

## Benchmark Evaluation

We have evaluated state-of-the-art visual object detection networks and HTD methods on the SPOD dataset.

### Object Detection Networks
- Testing conducted using the MMDetection framework
- Corresponding configuration files provided in our [SpecDETR](https://github.com/ZhaoxuLi123/SpecDETR) project for reproducible experiments

### HTD Methods
- Implemented using our previously developed HTD toolbox (planned for open-source release in the near future)

### Evaluation Results
Test results for SPOD, Avon, SanDiego and MUUFLGulfport datasets are available for download:
[Baidu Drive](https://pan.baidu.com/s/1lVtY5mPzhcovB89t_EjtnA?pwd=2789) (key:2789) or [OneDrive](https://dh7n-my.sharepoint.com/:u:/g/personal/vv_s3_tm9_site/EeqwBubB5o1KvC4Zu9kCLDEBRhUGPN1Ux25ZonXlk20GVQ)

After downloading, please unzip the file and place it in the `./eval_result/` directory. The folder structure is as follows:


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
Here is the folder structure description:

#### **HOD Results**  
- **`./XXXResult/HOD/result`**  
  - Stores test results from object detection networks (MMDetection output)  
  - Format: `.pkl` files  
- **`./XXXResult/HOD/cocojson`**  
  - Contains COCO-format detection results converted from `.pkl` files  
  - Format: `.json` files  

#### **HTD Results**  
- **`./XXXResult/HTD/result`**  
  - Stores detection score maps from HTD methods  
  - Format: `.mat` files (each file contains results for one test image)  
  - Shape: `Number of Target Classes × Height × Width`  
- **`./XXXResult/HTD/cocojson`**  
  - Contains COCO-format detection results converted from HTD outputs  
  - Format: `.json` files  

#### **Evaluation & Visualization**  
- **`./XXXResult/eval`**  
  - Stores quantitative evaluation metrics
  - Format: `.txt` files  
- **`./XXXResult/img`**  
  - Stores visualization results  
  - Formats: `.png`, `.eps`, or `.pdf`  


### Partial Quantitative Results of Visual Object Detection Networks on SPOD Dataset


|       Method       |   Backbone       | Image Size |       mAP50:95        |        mAP25        |        mAP50        |          mAP75        |      FLOPs       |        Params    |
|:------------------:|:------------:|:----------:|:---------------------:|:-------------------:|:-------------------:|:---------------------:|:---------------------:|:---------------:|
|  **Faster R-CNN**  | **ResNet50** |     x4     |     0.197    |   0.377    |    0.374    |     0.179     |     68.8G     | 41.5M   |
|  **Faster R-CNN**  | **RegNetX** |     x4     |     0.227    |   0.379   |    0.378    |    0.242    |     57.7G     | 31.6M   |
|  **Faster R-CNN**  | **ResNeSt50** |     x4     |     0.246    |   0.316    |    0.316    |     0.277     |     185.1G     | 44.6M   |
|  **Faster R-CNN**  | **ResNeXt101** |     x4     |     0.220    |   0.368   |    0.366   |     0.231     |     128.4G     | 99.4M   |
|  **Faster R-CNN**  | **HRNet** |     x4     |     0.320    |   0.404    |    0.402   |     0.345     |     104.4G     | 63.2M   |
|      **TOOD**      | **ResNeXt101** |     x4     |     0.304    |   0.464    |    0.440    |     0.303    |     114.3G     | 97.7M   |
| **CentripetalNet** | **HourglassNet104** |     x4     |    0.695    |   0.829    |   0.805    |     0.673     |     501.3G     | 205.9M   |
|   **CornerNet**    | **HourglassNet104** |     x4     |     0.626    |   0.736   |    0.712    |     0.609     |    462.6G     | 201.1M   |
|   **RepPoints**    | **ResNet50** |     x4     |     0.207   |   0.691   |    0.572   |     0.074     |     54.1G     | 36.9M   | |
|   **RepPoints**    | **ResNeXt101** |     x4     |   0.485    |   0.806    |    0.790   |     0.540     |     75.0G     | 58.1M   |
|   **RetinaNet**    | **EfficientNet** |     x4     |    0.462   |  0.836    |    0.811   |     0.466     |     36.1G     | 18.5M   |
|   **RetinaNet**    | **PVTv2-B3** |     x4     |   0.426    |  0.757   |    0.734    |    0.442     |     71.3G     | 52.4M   |
| **DeformableDETR** | **ResNet50** |     x4     |     0.231    |   0.692    |    0.560    |     0.147     |     58.7G     | 41.2M   |
|      **DINO**      | **ResNet50** |     x4     |    0.168   |   0.491   |    0.418   |    0.097    |    86.3G     | 47.6M   |
|      **DINO**      | **Swin-L** |     x4     |    0.757    |   0.852   |    0.842    |    0.764     |     203.9G     | 218.3M   |
|    **SpecDETR**    | -- |     x1     |     **0.856**    |  **0.938**    |    **0.930**   |     **0.863**   |     139.7G     | 16.1M   |



### Partial Quantitative Results of HTD Methods on SPOD Dataset


|  Method    |     mAP50:95     |   mAP25   |        mAP50        |          mAP75        |
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


For complete evaluation results, please refer to the files in: **`./eval_result/SPODResult/eval/`**

<br>


##  Training and Inference of the HOD Task

The [SpecDETR](https://github.com/ZhaoxuLi123/SpecDETR) project is built upon MMDetection project and supports training and inference of mainstream object detection networks on the SPOD dataset.

Configuration files used in the experiments are provided in the [SpecDETR](https://github.com/ZhaoxuLi123/SpecDETR)`/configs/VisualObjectDetectionNetwork/` folder.

The [SpecDETR](https://github.com/ZhaoxuLi123/SpecDETR) project also supports training and inference on HOD datasets generated by [HTDDataset2HODDataset.py](HTDDataset2HODDataset.py). The following modifications are required:

- Add object categories for HOD datasets in [SpecDETR](https://github.com/ZhaoxuLi123/SpecDETR)`/mmdet/datasets/hsi/HSI.py`:

  ```python
    class HSIDataset(CocoDataset):
        METAINFO = {
                'classes':
                ('CB',  'add new class',  'add new class'),
                'palette':
                [(220, 20, 60), (add new color),(add new color)]
                }
  ```
- Modify the dataset configuration file at [SpecDETR](https://github.com/ZhaoxuLi123/SpecDETR)`/configs/VisualObjectDetectionNetwork/_base_/datasets/hsi_detection4x.py` and `configs/_base_/datasets/hsi_detection.py`
  ```python
    data_root = 'new_HOD_dataset_path'   # ← Update to the local path of the new dataset
    train_pipeline = [
        dict(type='LoadHyperspectralImageFromFiles', normalized_basis=5000),  # Normalized_basis is the normalization constant in the paper. Update to match the new dataset. 
                        ]
  ```
  
-  To adapt the model for different hyperspectral datasets, you need to modify two critical parameters in the configuration file:

   1. **Input Channel Count** (number of spectral bands)
   2. **Number of Target Classes**

   Example modification for [SpecDETR](https://github.com/ZhaoxuLi123/SpecDETR)`/configs/VisualObjectDetectionNetwork/dino-5scale_swin-l-100e_hsi4x.py`:

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

Run  `SpecDETR/train.py` to train the model.

Run  `SpecDETR/test.py` to evaluate the model.

<br>

## HTD Result Conversion  

The script [HTDResult2Json.py](HTDResult2Json.py) converts detection score maps from traditional HTD methods into object-level prediction bounding box results (in JSON format) for object detection evaluation.  

<br>

## MMDetection Result Conversion  

The script [MMDdetResult2Json.py](MMDdetResult2Json.py) converts MMDetection's output `.pkl` result files into JSON format compatible with HODToolbox evaluation.  


<br>

## Quantitative Evaluation of Results  

The script [ResultEval.py](ResultEval.py) performs comprehensive evaluation of detection results, supporting both standard COCO metrics and specialized small target detection metrics.

1. **Standard COCO Metrics**:
   - Average Precision (AP) @ IoU thresholds [0.5:0.95]
   - AP@25, AP@50, AP@75
   - Average Recall (AR) @ IoU thresholds [0.5:0.95]
   - Recall@25, Recall@50, Recall@75

2. **Small Target-Specific Metrics**:
   - Precision/Recall/F1-score
   - False Alarm Rate (FAR)

<br>

## Results Visualization

The [ResultShow.py](ResultShow.py) script generates comprehensive visualizations for detection results analysis.


<br>

## Citation
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

## Contact

For any questions or inquiries regarding the HODToolbox or SpecDETR project, please feel free to contact lizhaoxu@nudt.edu.cn

