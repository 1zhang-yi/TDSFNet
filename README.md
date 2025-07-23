# Tensor Decomposition-based Subspace Fusion Network(TDSFNet)
## Introduction
TDSFNet is the accompanying code repository for the paper titled, "Tensor Decomposition-based Subspace Fusion Network for Multi-modal Medical Image Classification".

The goal of the repository is to provide an implementation of the TDSFNet and replicate the experiments in the paper.

Paper Link: https://ieeexplore.ieee.org/document/10902175
## Requirements
- Python-3.9.0
- CUDA-12.8
- Pytorch-2.7.1+cu126

## Data Preparation
We use three multi-modal medical image datasets.
- Pleural Effusion Cell(PEC) dataset: a multi-modal cell image set consisting of 400 HAL stained(HAL) cell images and 400 unstained original(orig) cell images. We made our datasets avaiable at:
- Age-related Macular Degeneration(AMD) dataset: https://arxiv.org/abs/2012.01879
- Seven-Point Checklist(SPC) dataset: https://ieeexplore.ieee.org/document/8333693


  
## Model Training
Training Model:
`python main.py`  



