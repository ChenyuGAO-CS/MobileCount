# MobileCount: An efficient encoder-decoder framework for real-time crowd counting

---

This repo is the official implementation of paper: "[MobileCount: An efficient encoder-decoder framework for real-time crowd counting.](https://www.sciencedirect.com/science/article/pii/S0925231220308912)" 

The code is developed based on C^3 Framework.


## Features
- **Convenient development kit**. It is a convenient dev kit on the six maintream datasets.
- **Solid baselines**. It provides some baselines of some classic pre-trained models, such as AlexNet, VGG, ResNet and so on. Base on it, you can easily compare your proposed models' effects with them.
- **Powerful log**. It does not only record the loss, visualization in Tensorboard, but also save the current code package (including parameters settings). The saved code package can be directly ran to reproduce the experiments at any time. You won't be bothered by forgetting the confused parameters.


## Performance
Due to limited spare time and the number of GPUs, I do not plan to conduct some experiments (named as "TBD"). If you are interested in the project, you are welcomed to submit your own experimental parameters and results. GCC(rd,cc,cl) stand for GCC dataset using **r**an**d**om/**c**ross-**c**amera/**c**ross-**l**ocation/ splitting, respectively.


|          Method          | UCF-QNRF  |   SHT A   |  SHT B  | WE |   UCF50   |
|--------------------------|-----------|-----------|---------|----|-----------|
| MobileCount              |131.1/222.6| 89.4/146.0| 9.0/15.4|11.1|284.8/293.8|
| MobileCount (x1.25)      |124.5/207.6| 82.9/137.9| 8.2/13.2|11.1|283.1/382.6|
| MobileCount (x2)         |117.9/207.5| 81.4/133.3| 8.1/12.7|11.5|284.5/421.2|


### data processing code
- [x] GCC
- [x] UCF-QNRF
- [x] ShanghaiTech Part_A
- [x] ShanghaiTech Part_B
- [x] WorldExpo'10
- [x] UCF_CC_50
- [x] UCSD
- [x] Mall

## Getting Started

### Preparation
- python 2.7
- pyTorch 1.0
  - Pytorch 1.0 (some networks only support 0.4): http://pytorch.org .
  - other libs in ```requirements.txt```, run ```pip install -r requirements.txt```.


- Installation
  - Clone this repo:
    ```
    git clone https://github.com/SelinaFelton/MobileCount.git
    ```

- Data Preparation
  - In ```./datasets/XXX/readme.md```, download our processed dataset or run the ```prepare_XXX.m/.py``` to generate the desity maps. If you want to directly download all processeed data (including Shanghai Tech, UCF-QNRF, UCF_CC_50 and WorldExpo'10), please visit the [**link**](https://mailnwpueducn-my.sharepoint.com/:f:/g/personal/gjy3035_mail_nwpu_edu_cn/EkxvOVJBVuxPsu75YfYhv9UBKRFNP7WgLdxXFMSeHGhXjQ?e=IdyAzA).
  - Place the processed data to ```../ProcessedData```.

- Folder Tree

    ```
    +-- MobileCount
    |   +-- datasets
    |   +-- misc
    |   +-- ......
    +-- ProcessedData
    |   +-- shanghaitech_part_A
    |   +-- ......
    ```
    

### Training

- set the parameters in ```config.py``` and ```./datasets/XXX/setting.py``` (if you want to reproduce our results, you are recommonded to use our parameters in ```./results_reports```).
- run ```python train.py```.
- run ```tensorboard --logdir=exp --port=6006```.

### Testing

We only provide an example to test the model on the test set. You may need to modify it to test your own models.


## Tips

In this code, the validation is directly on the test set. Strictly speaking, it should be evaluated on the val set (randomly selected from the training set, which is adopted in the paper). Here, for a comparable reproduction (namely fixed splitting sets), this code directly adopts the test set for validation, which causes that the results of this code are better than that of our paper. If you use this repo for academic research, you need to select 10% training data (or other value) as validation set. 

## Citation
If you find this project is useful for your research, please cite:
```
@article{wang2020mobilecount,
  title={MobileCount: An efficient encoder-decoder framework for real-time crowd counting},
  author={Wang, Peng and Gao, Chenyu and Wang, Yang and Li, Hui and Gao, Ye},
  journal={Neurocomputing},
  volume={407},
  pages={292--299},
  year={2020},
  publisher={Elsevier}
}
```
```
@article{gao2019c,
  title={C$^3$ Framework: An Open-source PyTorch Code for Crowd Counting},
  author={Gao, Junyu and Lin, Wei and Zhao, Bin and Wang, Dong and Gao, Chenyu and Wen, Jun},
  journal={arXiv preprint arXiv:1907.02724},
  year={2019}
}
```
