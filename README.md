# Tensor Decomposition-based Subspace Fusion Network(TDSFNet)
## Introduction
TDSFNet is the accompanying code repository for the paper titled, "Tensor Decomposition-based Subspace Fusion Network for Multi-modal Medical Image Classification".

The goal of the repository is to provide an implementation of the TDSFNet and replicate the experiments in the paper.

Paper Link:
## Requirements
- Python-
- CUDA-
- Pytorch-
- Other Packages `pip install -r requirements.txt`  

## Data Preparation
We use three multi-modal medical image datasets.
- Pleural Effusion Cell(PEC) dataset: a multi-modal cell image set consisting of 400 HAL stained(HAL) cell images and 400 unstained original(orig) cell images. We made our datasets avaiable at:
- Age-related Macular Degeneration(AMD) dataset: https://arxiv.org/abs/2012.01879
- Seven-Point Checklist(SPC) dataset: https://ieeexplore.ieee.org/document/8333693

### Data Organization(PEC)
├── cell                 
│   ├── HAL
│       ├── test
│           ├── HAL
│           ├── Met 5A
│           ├── WBC
│           ├── RBC
│       ├── train
│       ├── val
│   ├── orig
│       ├── test
│           ├── HAL
│           ├── Met 5A
│           ├── WBC
│           ├── RBC
│       ├── train
│       ├── val

  
## Model Training
train.py provides the training pipeline for PEC datasets.
The following paths need to be set in Config.py.
- save_path: Location to save model checkpoints.
- HAL_train, orig_train
- HAL_val, orig_val
- HAL_test, orig_test

Training Model:
`python train.py`  

We provide the trained TDSFNet checkpoint at 


