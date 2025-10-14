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

def plot_trim_HOSVD(r,m,n_dim,num_high, itr = 150,num_samples = 5,mu=1):
    
    n = n_dim[0]*n_dim[1]*n_dim[2]
    
    data_array = np.zeros((num_samples, itr, len(num_high)))

    for i in range(len(num_high)):
    
        for samples in range(num_samples):
        
            X = random_low_rank_HOSVD(n_dim,r)
            x = vectorize_np(X)
            A = gaussian_mx(m,n)
            b = A@x 
        
            y_IHT,error_IHT = adaptive_TIHT(A,b,X,r,lamda = 1/m,mu = mu,itr=itr, numb_high = num_high[i])

            data_array[samples,:,i] = error_IHT

    ### Getting plot data from this 

    median_data = np.zeros((itr, len(num_high)))

    for i in range(len(num_high)):
    
        median_data[:, i] = np.median(data_array[:,:, i],axis = 0)
    
    final_error = median_data[itr-1,]
    c = np.argmin(final_error)
    if median_data[itr-1,c] == np.inf:
        return median_data,[1,1]
    else:
        return median_data,num_high[c]

def plot_trim_HOSVD_fs(r,m,n_dim,num_high, itr = 150,num_samples = 5,mu=1):
    
    n= n_dim[0]*n_dim[1]*n_dim[2]
    
    data_array = np.zeros((num_samples, itr, len(num_high)))

    for i in range(len(num_high)):
    
        for samples in range(num_samples):
        
            X = random_low_rank_HOSVD(n_dim,r)
            x = vectorize_np(X)
            A_1 = np.random.normal(0.0,1.0,(n_dim[0],m))
            A_2 = np.random.normal(0.0,1.0,(n_dim[1],m))
            A_3 = np.random.normal(0.0,1.0,(n_dim[2],m))
            A_4 = linalg.khatri_rao(A_1,linalg.khatri_rao(A_2,A_3))
            A = A_4.T
            b = A@x 
        
            y_IHT,error_IHT = adaptive_TIHT(A,b,X,r,lamda = 1/m,mu = mu,itr=itr, numb_high = num_high[i])

            data_array[samples,:,i] = error_IHT

    ### Getting plot data from this 

    median_data = np.zeros((itr, len(num_high)))

    for i in range(len(num_high)):
    
        median_data[:, i] = np.median(data_array[:,:, i],axis = 0)
    
    final_error = median_data[itr-1,]
    c = np.argmin(final_error)
    if median_data[itr-1,c] == np.inf:
        return median_data,[1,1]
    else:
        return median_data,num_high[c]

def plot_trim_CP(r,m,n_dim,num_high, itr = 150,num_samples = 5, mu = 1):
    
    n= n_dim[0]*n_dim[1]*n_dim[2]
    
    data_array = np.zeros((num_samples, itr, len(num_high)))

    for i in range(len(num_high)):
    
        for samples in range(num_samples):
        
            X = random_low_rank_CP(n_dim,r)
            x = vectorize_np(X)
            A = gaussian_mx(m,n)
            b = A@x 
        
        y_IHT,error_IHT = adaptive_TIHT_CP(A,b,X,r,lamda = 1/m,mu = 1,itr=itr, numb_high = num_high[i])

        data_array[samples,:,i] = error_IHT

    ### Getting plot data from this 

    median_data = np.zeros((itr, len(num_high)))

    for i in range(len(num_high)):
    
        median_data[:, i] = np.median(data_array[:,:, i],axis = 0)
        
    final_error = median_data[itr-1,]
    c = np.argmin(final_error)
    if median_data[itr-1,c] == np.inf:
        return median_data,[1,1]
    else:
        return median_data,num_high[c]

def plot_trim_CP_fs(r,m,n_dim,num_high, itr = 150,num_samples = 5, mu = 1):
    
    n= n_dim[0]*n_dim[1]*n_dim[2]
    
    data_array = np.zeros((num_samples, itr, len(num_high)))

    for i in range(len(num_high)):
    
        for samples in range(num_samples):
        
            X = random_low_rank_CP(n_dim,r)
            x = vectorize_np(X)
            A_1 = np.random.normal(0.0,1.0,(n_dim[0],m))
            A_2 = np.random.normal(0.0,1.0,(n_dim[1],m))
            A_3 = np.random.normal(0.0,1.0,(n_dim[2],m))
            A_4 = linalg.khatri_rao(A_1,linalg.khatri_rao(A_2,A_3))
            A = A_4.T
            b = A@x 
        
        y_IHT,error_IHT = adaptive_TIHT_CP(A,b,X,r,lamda = 1/m,mu = mu,itr=itr, numb_high = num_high[i])

        data_array[samples,:,i] = error_IHT

    ### Getting plot data from this 

    median_data = np.zeros((itr, len(num_high)))

    for i in range(len(num_high)):
    
        median_data[:, i] = np.median(data_array[:,:, i],axis = 0)
        
    final_error = median_data[itr-1,]
    c = np.argmin(final_error)
    if median_data[itr-1,c] == np.inf:
        return median_data,[1,1]
    else:
        return median_data,num_high[c]
