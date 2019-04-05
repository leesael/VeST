# VeST: Very Sparse Tucker Factorization of Large-Scale Tensors 

## Overview
VeST is a tensor factorization method for partially observable data to output a very sparse core tensor and factor matrices.
VeST performs initial decomposition, determines unimportant entries in the decomposition results, removes the unimportant entries, and carefully updates the remaining entries.
To determine unimportant entries, we define and use entry-wise **responsibility** for the decomposed results.
The entries are updated iteratively in a coordinate descent manner in parallel for scalable computation.

## Paper
Please use the following citation for VeST:

Park, M., Jang, J.  & Sael, L. (2019). **VeST: Very Sparse Tucker Factorization of Large-Scale Tensors.**  arXiv:1904.02603 [cs.NA]
[[Paper](https://arxiv.org/abs/1904.02603)] [[Supplementary Material](/paper/supp-material.pdf)]

## Code
The source codes used in the paper are available. 
* VeST-v1.0: [download](/src/)

## Comparison
![compy_img](/img/Fig2.png)

## Dataset
| Name | Order | Dimensionality | Number of Entries | Download |
| --- | --- | --- | --- | --- |
| Yelp-s | 3 | 50 × 50 × 10 | 267 | [download](/sample/Yelp-s.zip) |
| AmzonFood-s | 3 | 50 × 50 × 10 | 495 | [download](/sample/AmzonFood-s.zip) |

## Discovery
We evaluated interpretability of VeST by investigating the factorization results of [MovieLens](https://grouplens.org/datasets/movielens/) dataset and visually showing that the sparse results enhance interpretability.

![discovery_img](/img/discovery.png)

## People
MoonJeoung Park (Daegu Gyeongnuk Institute of Science and Technology)  
[Jung-Gi Jang](https://datalab.snu.ac.kr/~jkjang) (Seoul National University)  
[Lee Sael](https://leesael.github.io/) (Seoul National University)
