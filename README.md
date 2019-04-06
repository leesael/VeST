# VeST: Very Sparse Tucker Factorization of Large-Scale Tensors 

## Overview
Given a large tensor, how can we decompose it to sparse core tensor
and factor matrices such that it is easier to interpret the results? How can we
do this without reducing the accuracy? Existing approaches either output dense
results or give low accuracy. In this paper, we propose VeST, a tensor factorization
method for partially observable data to output a very sparse core tensor and
factor matrices. VeST performs initial decomposition, determines unimportant
entries in the decomposition results, removes the unimportant entries, and updates
the remaining entries. To determine unimportant entries, we define and use
entry-wise **responsibility** for the decomposed results. The entries are updated iteratively
using a carefully derived coordinate descent rule in parallel for scalable
computation. VeST also includes an auto-search algorithm to give a good tradeoff
between sparsity and accuracy. Extensive experiments show that our method
VEST is at least 2:2 times sparser and at least 2:8 times more accurate compared
to competitors. Moreover, VeST is scalable in terms of dimensionality, number
of observable entries, and number of threads. Thanks to VeST, we successfully
interpret the decomposition result of real-world tensor data based on the sparsity
pattern of the factor matrices.

## Paper
Please use the following citation for VeST:

Park, M., Jang, J.  & Sael, L. (2019). **VeST: Very Sparse Tucker Factorization of Large-Scale Tensors.**  arXiv:1904.02603 [cs.NA]
[[Paper](https://arxiv.org/abs/1904.02603)] [[Supplementary Material](/paper/supp-main.pdf)]

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
