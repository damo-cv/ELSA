# ELSA: Enhanced Local Self-Attention for Vision Transformer

By [Jingkai Zhou](http://thefoxofsky.github.io/), [Pichao Wang](https://wangpichao.github.io/)\*, 
  Fan Wang, [Qiong Liu](https://www2.scut.edu.cn/sse/2018/0615/c16788a270756/page.htm)\*, Hao Li, [Rong Jin](http://www.cse.msu.edu/~rongjin/)

This repo is the official implementation of ["ELSA: Enhanced Local Self-Attention for Vision Transformer"](https://arxiv.org/abs/2112.12786).

**A FAST VERSION OF ELSA-SWIN COMING SOON!!**

## Introduction

<div align="center">
<img src=http://thefoxofsky.github.io/images/local_comp.png width=60%/>
</div>

Self-attention is powerful in modeling long-range dependencies, but it is weak in local finer-level feature learning. 
As shown in Figure 1, the performance of local self-attention (LSA) is just on par with convolution and inferior to 
dynamic filters, which puzzles researchers on whether to use LSA or its counterparts, which one is better, and what 
makes LSA mediocre. In this work, we comprehensively investigate LSA and its counterparts. We find that the devil lies 
in the generation and application of spatial attention. 

<div align="center">
<img src=http://thefoxofsky.github.io/images/elsa.png width=50%/>
</div>

Based on these findings, we propose the enhanced local self-attention (ELSA) with Hadamard attention and the ghost head, 
as illustrated in Figure 2. Experiments demonstrate the effectiveness of ELSA. Without architecture / hyperparameter 
modification, The use of ELSA in drop-in replacement boosts baseline methods consistently in both upstream and 
downstream tasks.

Please refer to our [paper](https://arxiv.org/abs/2112.12786) for more details.



## Model zoo

### ImageNet Classification

| Model       | #Params |   Pretrain  | Resolution | Top1 Acc | Download | 
| :---        |  :---:  |    :---:    |    :---:   |   :---:  |  :---:   |
| ELSA-Swin-T | 29M     | ImageNet 1K |     224    | 82.7     | [google](https://drive.google.com/file/d/1eM0FsRNEDX-NncIEfvl4yXOAJawy2Ls0/view?usp=sharing) / [baidu](https://pan.baidu.com/s/16lPWTybCeoHT4BMDaKDTYw?pwd=cw25) |
| ELSA-Swin-S | 53M     | ImageNet 1K |     224    | 83.5     | [google](https://drive.google.com/file/d/186PDbqrt2hEg8r5aH45D5D6bO-EWJgcO/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1qTyCm7vLXqd9KEMyIsKLrQ?pwd=e6b2) |
| ELSA-Swin-B | 93M     | ImageNet 1K |     224    | 84.0     | [google](https://drive.google.com/file/d/1J42asBqLb6iiKaYQaeoS2UpT7kUSjBr9/view?usp=sharing) / [baidu](https://pan.baidu.com/s/11V_IdRXXPo4IaqghUdLNCQ?pwd=3r11) |

### COCO Object Detection

| Backbone | Method | Pretrain | Lr Schd | Box mAP | Mask mAP | #Params | Download |
| :---:  | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ELSA-Swin-T | Mask R-CNN | ImageNet-1K | 1x | 45.7 | 41.1 | 49M | [google](https://drive.google.com/file/d/15wvHHwktc9Bzqro8SSrTTjKN_cWy8pLc/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1JlrYUs2SOPFrPRTNbsh44A?pwd=1z3c) |
| ELSA-Swin-T | Mask R-CNN | ImageNet-1K | 3x | 47.5 | 42.7 | 49M | [google](https://drive.google.com/file/d/1EOr-lLVTrci2A4qsD2P3wp-IVs27VVLm/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1IW-oonGGK8bDMdPqxNSt6g?pwd=7fzj) |
| ELSA-Swin-S | Mask R-CNN | ImageNet-1K | 1x | 48.3 | 43.0 | 72M | [google](https://drive.google.com/file/d/1pk185BpCGIDrUc1sAezjrA-wuWr_lYII/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1YFgpzvTK6MxkmqUdqt_HOQ?pwd=baiv) |
| ELSA-Swin-S | Mask R-CNN | ImageNet-1K | 3x | 49.2 | 43.6 | 72M | [google](https://drive.google.com/file/d/1akDKhHmh3_1PfB1Dds9YrpODPsx58-cc/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1QxcXlbz48jFIoMgfdlXMtw?pwd=9qc6) |
| ELSA-Swin-T | Cascade Mask R-CNN | ImageNet-1K | 1x | 49.8 | 43.0 | 86M | [google](https://drive.google.com/file/d/1wjw3lRqe1T_ph824RSb8Hk_yZ49d3CNI/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1C9pAXA2EUgv5twaaC_fQ6g?pwd=p85s) |
| ELSA-Swin-T | Cascade Mask R-CNN | ImageNet-1K | 3x | 51.0 | 44.2 | 86M | [google](https://drive.google.com/file/d/1B7sUVuGXZAzgZd0ud0MBsSlDbM8DkvC-/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1oQWh-jGB75NOUaHBMqJTWQ?pwd=8v7r) |
| ELSA-Swin-S | Cascade Mask R-CNN | ImageNet-1K | 1x | 51.6 | 44.4 | 110M | [google](https://drive.google.com/file/d/1WI4za90sXu4wv5rx_dTRm2X7xe30anil/view?usp=sharing) / [baidu](https://pan.baidu.com/s/18Wn79JvQnwiHfvD8_nELUg?pwd=qc8i) |
| ELSA-Swin-S | Cascade Mask R-CNN | ImageNet-1K | 3x | 52.3 | 45.2 | 110M | [google](https://drive.google.com/file/d/1mkhdwjyScpiBbobWc4TibtoYX4OLk9pp/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1I5wEDJz8sBdhukSXUEPOYQ?pwd=kxd1) |

### ADE20K Semantic Segmentation

| Backbone | Method  | Pretrain    | Crop Size | Lr Schd | mIoU (ms+flip) | #Params | Download    |
| :---:    | :---:   | :---:       | :---:     | :---:   | :---:   | :---:   | :---:       |
| ELSA-Swin-T | UPerNet | ImageNet-1K | 512x512 | 160K | 47.9 | 61M | [google](https://drive.google.com/file/d/1SjHyXNv-ODGsDxcDbvDPNHUQUoMq8FWR/view?usp=sharing) / [baidu](https://pan.baidu.com/s/13tAJq5Fw23Uzd-Sa-7wJTg?pwd=erxh) |
| ELSA-Swin-S | UperNet | ImageNet-1K | 512x512 | 160K | 50.4 | 85M | [google](https://drive.google.com/file/d/1RIaq45wBW0wnJlCK9cf-PDKGV41ECQ86/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1L4QEYoRc1_Veu4L-_jYVLQ?pwd=p84r) |

## Install

- Clone this repo:
```bash
git clone https://github.com/damo-cv/ELSA.git elsa
cd elsa
```

- Create a conda virtual environment and activate it:
```bash
conda create -n elsa python=3.7 -y
conda activate elsa
```

- Install `PyTorch==1.8.0` and `torchvision==0.9.0` with `CUDA==10.1`:
```bash
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.1 -c pytorch
```

- Install `CUDA==10.1` with `cudnn7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

- Install `Apex`:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../
```

- Install `mmcv-full==1.3.0`
```bash
pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.8.0/index.html
```

- Install other requirements:
```bash
pip install -r requirements.txt
```

- Install mmdet and mmseg:
```bash
cd ./det
pip install -v -e .
cd ../seg
pip install -v -e .
cd ../
```

- Build the elsa operation:
```bash
cd ./cls/models/elsa
python setup.py install
mv build/lib*/* .
cp *.so ../../../det/mmdet/models/backbones/elsa/
cp *.so ../../../seg/mmseg/models/backbones/elsa/
cd ../../../
```

## Data preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. Please prepare it under the following file structure:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```

Also, please prepare the [COCO](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md) 
and [ADE20K](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) datasets following their links. 
Then, please link them to `det/data` and `seg/data`.

## Evaluation

### ImageNet Classification

Run following scripts to evaluate pre-trained models on the ImageNet-1K:
```bash
cd cls

python validate.py <PATH_TO_IMAGENET> --model elsa_swin_tiny --checkpoint <CHECKPOINT_FILE> \
  --no-test-pool --apex-amp --img-size 224 -b 128

python validate.py <PATH_TO_IMAGENET> --model elsa_swin_small --checkpoint <CHECKPOINT_FILE> \
  --no-test-pool --apex-amp --img-size 224 -b 128

python validate.py <PATH_TO_IMAGENET> --model elsa_swin_base --checkpoint <CHECKPOINT_FILE> \
  --no-test-pool --apex-amp --img-size 224 -b 128 --use-ema
```

### COCO Detection

Run following scripts to evaluate a detector on the COCO:
```bash
cd det

# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

### ADE20K Semantic Segmentation

Run following scripts to evaluate a model on the ADE20K:
```bash
cd seg

# single-gpu testing
python tools/test.py <CONFIG_FILE> <SEG_CHECKPOINT_FILE> --aug-test --eval mIoU

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <SEG_CHECKPOINT_FILE> <GPU_NUM> --aug-test --eval mIoU
```

## Training from scratch

Due to randomness, the re-training results may have a gap of about 0.1~0.2% with the numbers in the paper.

### ImageNet Classification

Run following scripts to train classifiers on the ImageNet-1K:
```bash
cd cls

bash ./distributed_train.sh 8 <PATH_TO_IMAGENET> --model elsa_swin_tiny \
  --epochs 300 -b 128 -j 8 --opt adamw --lr 1e-3 --sched cosine --weight-decay 5e-2 \
  --warmup-epochs 20 --warmup-lr 1e-6 --min-lr 1e-5 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
  --mixup 0.8 --cutmix 1. --remode pixel --reprob 0.25 --clip-grad 5. --amp

bash ./distributed_train.sh 8 <PATH_TO_IMAGENET> --model elsa_swin_small \
  --epochs 300 -b 128 -j 8 --opt adamw --lr 1e-3 --sched cosine --weight-decay 5e-2 \
  --warmup-epochs 20 --warmup-lr 1e-6 --min-lr 1e-5 --drop-path 0.3 --aa rand-m9-mstd0.5-inc1 \
  --mixup 0.8 --cutmix 1. --remode pixel --reprob 0.25 --clip-grad 5. --amp

bash ./distributed_train.sh 8 <PATH_TO_IMAGENET> --model elsa_swin_base \
  --epochs 300 -b 128 -j 8 --opt adamw --lr 1e-3 --sched cosine --weight-decay 5e-2 \
  --warmup-epochs 20 --warmup-lr 1e-6 --min-lr 1e-5 --drop-path 0.5 --aa rand-m9-mstd0.5-inc1 \
  --mixup 0.8 --cutmix 1. --remode pixel --reprob 0.25 --clip-grad 5. --amp --model-ema
```

If GPU memory is not enough when training elsa_swin_base, you can use two nodes (2 * 8 GPUs), each with a batch size of 64 images/GPU.

### COCO Detection / ADE20K Semantic Segmentation

Run following scripts to train models on the COCO / ADE20K:
```bash
cd det 
# (or cd seg)

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```


## Acknowledgement

This work was supported by Alibaba Group through Alibaba Research Intern Program and the National Natural
Science Foundation of China (No.61976094).

Codebase from [pytorch-image-models](https://github.com/rwightman/pytorch-image-models),
              [ddfnet](https://github.com/theFoxofSky/ddfnet),
              [VOLO](https://github.com/sail-sg/volo),
              [Swin-Transformer](https://github.com/microsoft/Swin-Transformer),
              [Swin-Transformer-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection),
          and [Swin-Transformer-Semantic-Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation)


## Citing ELSA

```
@article{zhou2021ELSA,
  title={ELSA: Enhanced Local Self-Attention for Vision Transformer},
  author={Zhou, Jingkai and Wang, Pichao and Wang, Fan and Liu, Qiong and Li, Hao and Jin, Rong},
  journal={arXiv preprint arXiv:2112.12786},
  year={2021}
}
```
