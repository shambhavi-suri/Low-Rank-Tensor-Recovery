import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import rand, randn, choice, permutation

import torch

import tensorly as tl
from tensorly import decomposition
from tensorly.decomposition import parafac
from scipy import linalg


from KZTIHT_functions import *
from optimizing_trimming import *
from Adaptive_removal_functions import *

import time
import os
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
np.random.seed(idx*2)
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']

cols = []
itr = 150
n = 20*20*20*20
n_dim = [20,20,20,20]
m_choices = [int(n*0.02), int(n*0.03), int(n*0.04), int(n*0.07)]
num_choices = [500,500, 500, 600]

r = [2,2,2,2]
data_array_atiht = np.empty((itr,len(m_choices)))
#data_array_kztiht = np.empty((itr,len(m_choices)))
#data_array_tiht = np.empty((itr,len(m_choices)))
#time_array = np.empty((3, len(m_choices)))

for i in range(len(m_choices)):
    
    m = m_choices[i]
    num = num_choices[i]
    
    X = random_low_rank_HOSVD(n_dim,r)
    x = vectorize_np(X)
    A_0 = np.random.normal(0.0,1.0,(n_dim[0],m))
    A_1 = np.random.normal(0.0,1.0,(n_dim[1],m))
    A_2 = np.random.normal(0.0,1.0,(n_dim[2],m))
    A_3 = np.random.normal(0.0,1.0,(n_dim[3],m))
    A_4 = linalg.khatri_rao(A_1,linalg.khatri_rao(A_2,A_3))
    A_5 = linalg.khatri_rao(A_0,A_4) 
    A = A_5.T
    b = A@x 
    #start = time.time()
    #y_KZIHT, error_KZIHT = KZIHT_HOSVD_RR(A,b,X,n,r,gamma = 1, itr = itr)
    #kz_time = time.time() - start
    start = time.time()
    y_ATIHT, error_ATIHT = adaptive_TIHT(A,b,X,r,lamda = 1/m,mu = 1,itr=itr, numb_high = num)
    atiht_time = time.time() - start
    #start = time.time()
    #y_IHT_2,error_IHT_2 = TIHT_HOSVD(A,b,X,r,lamda=1/m,itr = itr)
    #tiht_time = time.time() - start
    data_array_atiht[:,i] = error_ATIHT
    #data_array_kztiht[:,i] = error_KZIHT
    #data_array_tiht[:,i] = error_IHT_2
    #time_array[0,i] = atiht_time
    #time_array[1,i] = kz_time
    #time_array[2,i] = tiht_time

data_array =  pd.DataFrame(data_array_atiht,columns = list(map(str, m_choices)))
data_array.to_csv(f"data_atiht_{idx}.csv")
#data_array =  pd.DataFrame(data_array_kztiht,columns = list(map(str, m_choices)))
#data_array.to_csv(f"data_kztiht_{idx}.csv")
#data_array =  pd.DataFrame(data_array_tiht,columns = list(map(str, m_choices)))
#data_array.to_csv(f"data_tiht_{idx}.csv")
#pd.DataFrame(time_array,columns = list(map(str, m_choices))).to_csv(f"time_{idx}.csv")
