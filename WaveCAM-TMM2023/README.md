# Wave-like Class Activation Map with Representation Fusion for Weakly-Supervised Semantic Segmentation
Pytorch implementation of "Wave-like Class Activation Map with Representation Fusion for Weakly-Supervised Semantic Segmentation". 

## Installation

Install dependencies:
```
conda env create -f environment.yml
```
Python 3.6, PyTorch 1.9, and others in environment.yml

## Data Preparation

Download  PASCAL VOC 2012 dataset:

```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar –xvf VOCtrainval_11-May-2012.tar
```

Download MS COCO 2014 dataset:

```
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip 
```

## Usage for Multi-Stage WSSS

```
cd multi_stage
```

### PASCAL VOC Dataset:**

**1. Train WaveCAM and generate Initialize pseudo labels.**

```
python run_wavecam_voc.py --voc12_root your_path --work_space your_work_space --train_cam_pass True --train_wavecam_pass True --make_wavecam_pass True --eval_cam_pass True
```

**2. Train IRN and generate pseudo labels.**

```
python run_wavecam_voc.py --voc12_root your_path --work_space your_work_space --cam_to_ir_label_pass True --train_irn_pass True --make_sem_seg_pass True --eval_sem_seg_pass True 
```
**3. Train the fully supervised semantic segmentation network.**

Please download [ImageNet pre-trained model](https://drive.google.com/file/d/14soMKDnIZ_crXQTlol9sNHVPozcQQpMn/view?usp=sharing) .

We refer to [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch) to train DeepLab-v2. Please replace the groundtruth mask with the generated pseudo mask when training.

### **MS COCO Dataset:**

**1. Train WaveCAM and generate Initialize pseudo labels.**

```
python run_wavecam_coco.py --mscoco_root your_path --work_space your_work_space --train_cam_pass True --train_wavecam_pass True --make_wavecam_pass True --eval_cam_pass True 
```

**2. Train IRN and generate pseudo labels.**

```
python run_wavecam_coco.py --mscoco_root your_path --work_space your_work_space --cam_to_ir_label_pass True --train_irn_pass True --make_sem_seg_pass True --eval_sem_seg_pass True 
```

**3. Train the fully supervised semantic segmentation network.**

The same as PASCAL VOC Dataset.

## Usage for End-to-End WSSS

```
cd end_to_end
```

Install dependencies:

```
pip install -r requirements.txt
```

Download  Pre-trained weights:

```
https://github.com/NVlabs/SegFormer
```

Training on VOC:

```
bash train/run_wavecam_voc.sh
```

Training on COCO:

```
bash train/run_wavecam_coco.sh
```

## Citation
If you find the paper or code useful, please consider citing:
```@article{xu2023wave,
  title={Wave-Like Class Activation Map With Representation Fusion for Weakly-Supervised Semantic Segmentation},
  author={Xu, Rongtao and Wang, Changwei and Xu, Shibiao and Meng, Weiliang and Zhang, Xiaopeng},
  journal={IEEE Transactions on Multimedia},
  year={2023},
  publisher={IEEE}
}
```

## License me 

**This code is only for reviewers to understand implementation details.**

This code is available only for non-commercial use.


## Thanks 

Some implementation code comes from：

ReCAM：https://github.com/zhaozhengChen/ReCAM

AFA：https://github.com/rulixiang/afa.git

DeepLab：https://github.com/kazuto1011/deeplab-pytorch




