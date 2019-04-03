# VeST: Very Sparse Tucker Factorization of Large-Scale Tensors 

## Overview
VeST is a tensor factorization method for partially observable data to output a very sparse core tensor and factor matrices.
VeST performs initial decomposition, determines unimportant entries in the decomposition results, removes the unimportant entries, and carefully updates the remaining entries.
To determine unimportant entries, we define and use entry-wise **responsibility** for the decomposed results.
The entries are updated iteratively in a coordinate descent manner in parallel for scalable computation.

## Paper
Please use the following citation for VeST:

Park, M., Jang, J.  & Sael, L. (2019). **VeST: Very Sparse Tucker Factorization of Large-Scale Tensors.** 
[[Paper]()] [[Supplementary Material](/paper/supp-material.pdf)]

## Code
The source codes used in the paper are available. 
* VeST-v1.0: [[download](/src/)]

## Comparison
![comp_img](/img/Fig2.png)

## Dataset


## People
[MoonJeoung Park] (Daegu Gyeongnuk Institute of Science and Technology)  
[Jung-Gi Jang](https://datalab.snu.ac.kr/~jkjang) (Seoul National University)  
[Lee Sael](https://leesael.github.io/) (Seoul National University)
