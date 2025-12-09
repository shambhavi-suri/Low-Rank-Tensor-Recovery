## On Trimming Tensor-structured Measurements and Efficient Low-rank Tensor Recovery

This repository contains the code for our experiments and plots for our paper - [On Trimming Tensor-structured Measurements and Efficient Low-rank Tensor Recovery](https://www.arxiv.org/abs/2502.02843).

## Organization:

* The files [Adaptive_removal_functions.py](https://github.com/shambhavi-suri/Low-Rank-Tensor-Recovery/tree/main/large_tensor_experiments) and [KZTIHT_functions.py](https://github.com/shambhavi-suri/Low-Rank-Tensor-Recovery/tree/main/large_tensor_experiments) have the code for the TrimTIHT and KaczTIHT algorithms respectively. 

* The [optimizing_trimming.py](https://github.com/shambhavi-suri/Low-Rank-Tensor-Recovery/blob/main/large_tensor_experiments/optimizing_trimming.py) file contains code for determining the best trimming parameter for both CP/HOSVD low rank tensors. This determines the best value by running multiple trials over random tensors of the given dimension.

## Replicating Code:


## Real-world Experiemnts:

The data for the candle video dataset can be found in the data file. The code for the analysis can be found in the [Candle Video Low rank Recovery.ipynb](https://github.com/shambhavi-suri/Low-Rank-Tensor-Recovery/blob/main/Jupyter_Notebooks/Candle%20Video%20Low%20rank%20Recovery.ipynb) Jupyter Notebook. 
