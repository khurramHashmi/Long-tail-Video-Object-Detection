# Long-tail-Video-Object-Detection


This repository experiments by extending the idea of long tail object detection from images into Video domain. The idea of long tail object detection is inspired by this [**Paper**](http://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Overcoming_Classifier_Imbalance_for_Long-Tail_Object_Detection_With_Balanced_Group_CVPR_2020_paper.pdf).


## Requirements

### 1. Environment:

- python 3.7
- pytorch 1.11
- torchvision 1.10
- mmtracking 0.12
- mmdet 2.23
- mmcv 1.4.8
- pycocotools 2.0.4


### 2. Data:
#### a. Dataset:
    
This model works with ImageNet VID dataset. Upload the dataset and link the path in 'configs/_base_/datasets/gs_imagenet_vid_fgfa_style.py' file.

#### b. Dataset annotations:

Only COCO annotation format is supported. Upload the annotation file and link the path in 'configs/_base_/datasets/gs_imagenet_vid_fgfa_style.py' file.

#### c. Pretrained models:

The pretrained model will be saved in 'work_dirs' directory after training.

#### d. Intermediate files:
Intermediate files are already present and linked in config file. Files are present in 'intermediate_files' directory. 

## Training

Use the following commands to train the model:


```train
# Single GPU

CUDA_VISIBLE_DEVICES=0 PORT=29501 ./tools/dist_train.sh configs/vid/fgfa/gs_fgfa_faster_rcnn_r50_dc5_1x_imagenetvid.py 1

# Multi GPU distributed training
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./tools/dist_train.sh configs/vid/fgfa/gs_fgfa_faster_rcnn_r50_dc5_1x_imagenetvid.py 4
```

> ***Important***: According to the Linear Scaling Rule, you need to set the learning rate proportional to the batch size if you use different GPUs or images per GPU, e.g., lr=0.01 for 4 GPUs * 2 img/gpu and lr=0.08 for 16 GPUs * 4 img/gpu.


## Testing

Use the following command to test the model:
```test
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/vid/fgfa/gs_fgfa_faster_rcnn_r50_dc5_1x_imagenetvid.py --checkpoint work_dirs/gs_fgfa_faster_rcnn_r50_dc5_1x_imagenetvid/latest.pth --eval bbox \
--out work_dirs/gs_fgfa_faster_rcnn_r50_dc5_1x_imagenetvid/gs_fgfa_faster_rcnn_r50_dc5_1x_imagenetvid.pkl
```

> - Filename of the output results in pickle format. If not specified, the results will not be saved to a file.
> - Items to be evaluated on the results. `bbox` for bounding box evaluation only.
> - The evaluation results will be shown in markdown table format.


## Results
The model acheives mAP@50 of 75.9.

## Credit
This code is largely based on [**mmtracking**](https://github.com/open-mmlab/mmtracking) and [**BalancedGroupSoftmax**](https://github.com/FishYuLi/BalancedGroupSoftmax).
