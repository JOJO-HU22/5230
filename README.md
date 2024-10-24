# Adversarial Stickers: A Stealthy Attack Method in the Physical World
This repository contains the code for Adversarial Stickers introduced in the following paper
[Adversarial Stickers: A Stealthy Attack Method in the Physical World](https://ieeexplore.ieee.org/abstract/document/9779913) (TPAMI 2022)
## Preparation

### Environment Settings:

This project is tested under the following environment settings:
+ OS: Ubuntu 18.04
+ GPU: Geforce 2080 Ti
+ Python: 3.8.11
+ PyTorch: 1.7.1+cu110
+ Torchvision: 0.8.2+cu110

### Data Preparation：
+ face
Please download the dataset ([LFW](http://vis-www.cs.umass.edu/lfw/), as mentioned in the paper， we should find the datasets in lfw that contains 5749 Folders with person's name， the correct link is http://vis-www.cs.umass.edu/lfw/lfw.tgz.

[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)) and place it in ```./datasets/```.

The directory structure example is:
```
datasets
-datasets name
 --person 1
   ---pic001
   ---pic002
   ---pic003  
```
+ stickers
Stickers used for adversarial attacks should be placed in the `./stickers/` directory. These stickers will be resized, rotated, and manipulated for attacks as specified in the `predict.py` script.

The directory structure should look like this:

stickers
 --sticker1.png
 --sticker2.png

Prepare the pre-defined stickers and place them in ```./stickers/```. In our project, we choose bs13.png as the final sticker.
```python
class Config(object):
    bound = 15  # Critical value of the gap between label1 and label2
    id_threat = 1  # ID of threat model in ['arcface', 'facenet', 'sphereface', 'cosface']
    sticker_name = 'bs13'  # Sticker used for the attack (Change the figure name here)
    scale = 12  # The scale of the sticker
```

### Model Preparation：
Tool models ([FaceNet](https://github.com/timesler/facenet-pytorch), [CosFace](https://github.com/deepinsight/insightface/tree/master/recognition), [SphereFace](https://github.com/clcarwin/sphereface\_pytorch)) should be placed in ```./models/```

The corresponding ```./utils/predict.py``` should be changed as needed.

### Other Necessary Tools:
+ Python tools for [3D face](https://github.com/YadiraF/face3d/tree/master/face3d)
+ BFM Data: ```./BFM/BFM.mat```
+ Shape predictor for face landmarks ([68](https://github.com/r4onlyrishabh/facial-detection/tree/master/dataset), [81](https://github.com/codeniko/shape_predictor_81_face_landmarks))

## Quick Start
Hyperparameter settings: ```./utils/config.py```

Running this command for attacks:

```
python attack_single.py
```

## Citation
If you find our methods useful, please consider citing:
```
@article{wei2022adversarial,
  title={Adversarial Sticker: A Stealthy Attack Method in the Physical World},
  author={Wei, Xingxing and Guo, Ying and Yu, Jie},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```
