# MCADFusion
MCADFusion: a novel multi-scale convolutional attention decomposition method for enhanced infrared and visible light image fusion
[Zixiang Zhao](https://github.com/DMengHao/MCADFusion)
-[*[Paper]*](https://www.aimspress.com/aimspress-data/era/2024/8/PDF/era-32-08-233.pdf)  
-[*[Supplementary Materials]*](https://www.aimspress.com/aimspress-data/era/2024/8/PDF/era-32-08-233.pdf)  

## Citation

```
@article{zhang2024mcadfusion,
  title={MCADFusion: a novel multi-scale convolutional attention decomposition method for enhanced infrared and visible light image fusion.},
  author={Zhang, Wangwei and Dai, Menghao and Zhou, Bin and Wang, Changhai},
  journal={Electronic Research Archive},
  volume={32},
  number={8},
  year={2024}
}
```

## Abstract

This paper presents a method called MCADFusion, a feature decomposition technique specifically designed for the fusion of infrared and visible images, incorporating target radiance and detailed texture. MCADFusion employs an innovative two-branch architecture that effectively extracts and decomposes both local and global features from different source images, thereby enhancing the processing of image feature information. The method begins with a multi-scale feature extraction module and a reconstructor module to obtain local and global feature information from rich source images. Subsequently, the local and global features of different source images are decomposed using the the channel attention module (CAM) and the spatial attention module (SAM). Feature fusion is then performed through a two-channel attention merging method. Finally, image reconstruction is achieved using the restormer module. During the training phase, MCADFusion employs a two-stage strategy to optimize the network parameters, resulting in high-quality fused images. Experimental results demonstrate that MCADFusion surpasses existing techniques in both subjective visual evaluation and objective assessment on publicly available TNO and MSRS datasets, underscoring its superiority.

## 🌐 Usage

### ⚙ Network Architecture

Our CDDFuse is implemented in ``net.py``.

### 🏊 Training
**1. Virtual Environment**
```
# create virtual environment
conda create -n cddfuse python=3.8.10
conda activate cddfuse
# select pytorch version yourself
# install cddfuse requirements
pip install -r requirements.txt
```

**2. Data Preparation**

Download the MSRS dataset from [this link](https://github.com/Linfeng-Tang/MSRS) and place it in the folder ``'./MSRS_train/'``.

**3. Pre-Processing**

Run 
```
python dataprocessing.py
``` 
and the processed training dataset is in ``'./data/MSRS_train_imgsize_128_stride_200.h5'``.

**4. MCADFusion Training**

Run 
```
python train.py
``` 
and the trained model is available in ``'./models/'``.

### 🏄 Testing

Pretrained models

**2. Test datasets**

The test datasets used in the paper have been stored in ``'./test_img/TNO'`` and ``'./test_img/MSRS'``.

If you want to infer with our CDDFuse and obtain the fusion results in our paper, please run 
```
python test_IVF.py
```
The output for ``'test_IVF.py'`` is:

```
================================================================================
The test result of TNO :
                 EN      SD        SF       SCD     Qabf    SSIM
CDDFuse         7.20    48.39    13.58     1.87     0.48    0.99
================================================================================

================================================================================
The test result of MSRS :
                 EN      SD      SF        SCD     Qabf    SSIM
CDDFuse         6.86    50.77   12.84     1.81    0.66    1.00
================================================================================
```

## 🙌 MCADFusion

### Illustration of our MCADFusion model.

<img src="FrameWork.pdf" width="90%" align=center />

## 📖 Related Work

Other work has not been disclosed, and the source code will continue to be disclosed subsequently.





