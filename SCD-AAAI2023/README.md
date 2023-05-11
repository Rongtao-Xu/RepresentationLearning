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

Before Running, build python extension module:
```
cd wrapper/bilateralfilter
swig -python -c++ bilateralfilter.i
python setup.py install
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

## Citation
If you find the paper or code useful, please consider citing:
```@article{xu2023self,
  title={Self Correspondence Distillation for End-to-End Weakly-Supervised Semantic Segmentation},
  author={Xu, Rongtao and Wang, Changwei and Sun, Jiaxi and Xu, Shibiao and Meng, Weiliang and Zhang, Xiaopeng},
  journal={arXiv preprint arXiv:2302.13765},
  year={2023}
}
```

### Thanks ###

Some implementation code comes from：

AFA：https://github.com/rulixiang/afa.git

SegFormer：https://github.com/NVlabs/SegFormer

1Stage：https://github.com/visinf/1-stage-wseg




