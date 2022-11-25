# Self Correspondence Distillation For End-to-End Weakly-Supervised Semantic Segmentation
Pytorch implementation of Self Correspondence Distillation For End-to-End Weakly-Supervised Semantic Segmentation.



### License me ###

This code is available only for non-commercial use.

### TSCD framework's details ###
|  Technique   |  Implementation Code |
|  ----  | ----  |
| **Self Correspondence Distillation** | scripts/dist_train_voc.py, utils/corrloss.py |
| **Variation-aware Refine Module** | networks/VARM.py |
| **TSCD framework** | scripts/dist_train_voc.py, networks/TSCD_model.py |

### Installation  ###

Install dependencies:
```
pip install -r requirements.txt
```
### Data Preparation  ###

Download VOC dataset:

```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar –xvf VOCtrainval_11-May-2012.tar
```

Download COCO dataset:

```
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip 
```

### Evaluation

Evaluation on VOC:

Please use the validate function in scripts/dist_train_voc.py, the trained model is in`./weight/tscd_model.pth` 

```
python scripts/dist_train_voc.py
```



### Training ###

Download  Pre-trained weights:

```
https://github.com/NVlabs/SegFormer
```

Training on VOC:
```
bash train/run_sbatch_attn_reg.sh
```


### Thanks ###

Some implementation code comes from：

AFA：https://github.com/rulixiang/afa.git

SegFormer：https://github.com/NVlabs/SegFormer

1Stage：https://github.com/visinf/1-stage-wseg




