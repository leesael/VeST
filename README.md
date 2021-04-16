# Large-Scale Tucker Tensor Factorization for Sparse and Accurate Decomposition

## Overview
Given a large tensor, how can we decompose it to sparse core tensor 
and factor matrices without reducing the accuracy?
Existing approaches either output dense results or have scalability issues.
In this paper, we propose VEST, a tensor factorization method for large partially observable data to output a very sparse core tensor and factor matrices.
VEST performs initial decomposition and iteratively determines unimportant entries in the decomposition results, removes the
unimportant entries, and updates the remaining entries.
To determine unimportant entries of factor matrices and core tensor,
we define and use entry-wise ‘responsibility’ of the current decomposition.
For scalable computation, the entries are updated iteratively using a carefully derived coordinate descent rule in parallel.
Also, VEST automatically searches for the best sparsity ratio that results in a balanced trade-off between sparsity and accuracy.
Extensive experiments show that our method VEST produces more accurate results compared to the best performing competitors for all tested real-life datasets.
Moreover, VEST is scalable in terms of dimensionality, number of observable entries, and number of threads.

## Paper
Please use the following citation for VeST:

Park, M., Jang, J.  & Sael, L. (2020). **VeST: Very Sparse Tucker Factorization of Large-Scale Tensors.**  2021 IEEE International Conference on
Big Data and Smart Computing (BigComp)
[[Paper](./paper/paper.pdf)] [[Supplementary Material](./paper/supplementary.pdf)]

## Comparison
![compy_img](/img/Fig2.png)

## People
MoonJeong Park (Pohang University of Science and Technology)  
[Jung-Gi Jang](https://datalab.snu.ac.kr/~jkjang) (Seoul National University)  
[Lee Sael](https://leesael.github.io/) (Ajou University)

## Code
The source codes used in the paper are available. 
* VeST-v1.0: [download](/src/)
