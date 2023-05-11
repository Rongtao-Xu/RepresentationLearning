<h2 align="center">RSSFormer: Foreground Saliency Enhancement for Remote Sensing Land-Cover Segmentation</h2>

 This repository contains the code for the paper RSSFormer: Foreground Saliency Enhancement for Remote Sensing Land-Cover Segmentation (TIP2023) 


## Getting Started


#### Requirements:
- pytorch >= 1.1.0
- python >=3.6

### Prepare LoveDA Dataset

```bash
ln -s </path/to/LoveDA> ./LoveDA
```


### Train Model
```bash 
bash ./scripts/train.sh
```
Evaluate on eval set 
```bash
bash ./scripts/eval.sh
```

## Citation
If you find the paper or code useful, please consider citing:
```@article{xu2023rssformer,
  title={Rssformer: Foreground saliency enhancement for remote sensing land-cover segmentation},
  author={Xu, Rongtao and Wang, Changwei and Zhang, Jiguang and Xu, Shibiao and Meng, Weiliang and Zhang, Xiaopeng},
  journal={IEEE Transactions on Image Processing},
  volume={32},
  pages={1052--1064},
  year={2023},
  publisher={IEEE}
}
```



