import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import rand, randn, choice, permutation

import torch

import tensorly as tl
from tensorly import decomposition
from tensorly.decomposition import parafac
from scipy import linalg

## Defining Measurement Matrices ##

def gaussian_mx(m,N):
    A = np.random.normal(0.0, 1.0, [m, N])
    return A

def hadamard_mx(m,N):
    A = hadamard(N)
    l = permutation(np.range(N))
    return A[l[:m],:]

def vectorize_tl(X): ##Vectorisation for tensors
    x=X.numpy()
    x=x.reshape(-1)
    return x

def vectorize_np(X):  ##Vecorisation for numpy arrays
    x=X.reshape(-1)
    return x

def row_normalised_mx(A,b,n_dim):
    B = np.empty_like(A)
    c = np.empty_like(b)
    for i in range(np.shape(A)[0]):
        B[i,:] = np.sqrt(n_dim)*A[i,:]/np.linalg.norm(A[i,:])
        c[i] = np.sqrt(n_dim)*b[i]/np.linalg.norm(A[i,:])
    return B,c

## Thresholding operators #########

def random_low_rank_HOSVD(n,r,eps = 0.1):
    C=np.random.normal(0,1,size=r)+eps
    C=tl.tensor(C)
    #C.shape
    X=C ##core tensor

    U=[]
    for i in range(len(n)):
        M=np.random.normal(0,1,size=(n[i],n[i]))+eps
        u,sigma,v=np.linalg.svd(M)
        U.append(u[:,0:r[i]])

    for i in range(len(n)):
        X=tl.tenalg.mode_dot(X,U[i],i)
    return X

def random_lowrank_HOSVD_tensor(n,r,eps = 2):
    C=np.random.normal(0,1,size=r)+eps*np.random.uniform(0,1,size=r)
    C=tl.tensor(C)
    #C.shape
    X=C ##core tensor

    U=[]
    for i in range(len(n)):
        M=np.random.normal(0,1,size=(n[i],n[i]))+eps
        u,sigma,v=np.linalg.svd(M)
        U.append(u[:,0:r[i]])

    for i in range(len(n)):
        X=tl.tenalg.mode_dot(X,U[i],i)
    return X


def random_low_rank_CP(n,r,eps = 0.1):   #### CP Rank r
    
    L = []
    for i in range(0,len(n)):
        C=np.random.normal(0,1,size=(n[i],r))+eps
        L = L + [C]
    
    X = np.zeros(n)
    for i in range(r):
        U_r = np.array(L[0])[:,i]
        for j in range(1, len(n)):
            prod = np.array(L[j])[:,i]
            U_r = np.multiply.outer(U_r,prod)
        X = X + U_r
        
    C=tl.tensor(X) #Changing data frame to tensor
    C.shape
    return C

def HOSVD_rank_app(tensor,r): ## HOSVD rank-r approximation
    
    core, factors = tl.decomposition.tucker(tensor.numpy(), r) #Decomposition function is used 
    answer = torch.tensor(tl.tucker_to_tensor([core, factors]))
    
    return answer

def CP_rank_app(tensor,r):  ## CP rank-r approximation
    
    factors = parafac(tl.tensor(tensor), rank=r)
    answer = tl.cp_to_tensor(factors)
    
    return answer

def TIHT_CP(AA,yy,X,r,lamda = 1, itr = 100): 
    
    n = np.shape(X)
    X_ravel = np.ravel(X)
    
    error = np.zeros(itr)
    
    vXX = torch.zeros(0) 
    converge = True
    j = 0
            
    while converge == True and j < itr:
        try:
            WW = np.array(vectorize_np(vXX)) + lamda*np.matmul(AA.T, (yy - np.matmul(AA, np.array(vectorize_np(vXX)))))
            WW = torch.reshape(torch.tensor(WW), n)
            vXX = CP_rank_app(WW,r)
            error[j] = np.linalg.norm(vectorize_np(vXX)- X_ravel)/np.linalg.norm(X_ravel)
            j = j+1
        
        except np.linalg.LinAlgError:
            print("Doesn't converge")
            y = np.zeros(np.shape(X_ravel)[0])-1
            error = np.zeros(itr)+np.inf
            converge = False
        
    return vXX, error

def TIHT_HOSVD(AA,yy,X,r,lamda = 1, itr = 100): 
    
    n = np.shape(X)
    X_ravel = np.ravel(X)
    
    error = np.zeros(itr)
    
    vXX = torch.zeros(n)
    converge = True
    k = 0
            
    while converge == True and k < itr:
        try:
                WW = np.array(vectorize_tl(vXX)) + lamda* np.matmul(AA.T, (yy - np.matmul(AA, np.array(vectorize_tl(vXX)))))
                WW = torch.reshape(torch.tensor(WW), n)
                vXX = HOSVD_rank_app(WW,r)
                error[k] = np.linalg.norm(vectorize_tl(vXX)- X_ravel)/np.linalg.norm(X_ravel)    
                k = k+1 
                
        except np.linalg.LinAlgError:
            print("Doesn't converge")
            y = np.zeros(np.shape(X_ravel)[0])-1
            error = np.zeros(itr)+np.inf
            converge = False
  
    return vXX, error

def KZIHT_RR(A,b,x,s,gamma=1,itr=100): ## Selecting rows with replacement, gamma-step size for Kaczmarz
    
    error = np.zeros(itr)
    m = np.shape(A)[0]
    y= np.zeros(np.shape(x)[0])
    
    for k in range(itr): # Outer iteration for IHT updates
        
        t = permutation(np.arange(m))
        
        for j in range(m): #Inner iteration for Kaczmarz updates
            
            a = A[t[j],:]
            y = y + gamma*(b[t[j]] - a@y)*a/(np.linalg.norm(a)**2)
                           
        y = sparse_vect(y,s)
        error[k] = np.linalg.norm(y-x)/np.linalg.norm(x)
        
    return y,error

def KZIHT_HOSVD_RR(A,b,X,n,r,gamma = 1, lamda = 1, itr = 100):
    
    error = np.zeros(itr)
    m = np.shape(A)[0]
    n_dim =  np.shape(A)[1]
    
    n = np.shape(X)
    x = np.ravel(X)
    
    A,b = row_normalised_mx(A,b,n_dim)
    
    y = np.zeros(np.shape(x)[0]) 
    converge = True
    k = 0
    
    gamma = gamma*n_dim/m
            
    while converge == True and k < itr:
        y_old = y
        try:
            t = permutation(np.arange(m))
            for j in range(m): #Inner iteration for Kaczmarz updates
                a = A[t[j],:]
                y = y + gamma*(b[t[j]] - a@y)*a/(np.linalg.norm(a)**2)    
            y = y_old + lamda*(y - y_old)   
            WW = torch.reshape(torch.tensor(y), n)
            y = vectorize_tl(HOSVD_rank_app(WW,r))
            error[k] = np.linalg.norm(vectorize_np(y)-x)/np.linalg.norm(x)
            k = k+1 
                
        except np.linalg.LinAlgError:
            print("Doesn't converge")
            y = np.zeros(np.shape(x)[0])-1
            error = np.zeros(itr)+np.inf
            converge = False
                             
    return y,error

def KZIHT_CP_RR(A,b,X,n,r,gamma = 1,lamda = 1, itr = 100):
    
    error = np.zeros(itr)
    m = np.shape(A)[0]
    n_dim =  np.shape(A)[1]
    
    n = np.shape(X)
    x = np.ravel(X)
    A,b = row_normalised_mx(A,b,n_dim)
    
    y = np.zeros(np.shape(x)[0])    
    converge = True
    k = 0
    
    gamma = gamma*n_dim/m
            
    while converge == True and k < itr:
        try:
            y_old = y
            t = permutation(np.arange(m))
            for j in range(m): #Inner iteration for Kaczmarz updates
                a = A[t[j],:]
                y = y + gamma*(b[t[j]] - a@y)*a/(np.linalg.norm(a)**2)     
                
            y = y_old + lamda*(y - y_old)
            WW = torch.reshape(torch.tensor(y), n)
            y = CP_rank_app(WW,r)
            error[k] = np.linalg.norm(vectorize_np(y)-x)/np.linalg.norm(x)
            y = vectorize_np(y)
            k = k+1 
                             
        except np.linalg.LinAlgError:
            print("Doesn't converge")
            y = np.zeros(np.shape(x)[0])-1
            error = np.zeros(itr)+np.inf
            converge = False
            
    return y,error

def KZPT_HOSVD_RR(A,b,X,n,r, period = 1,gamma = 1, lamda = 1, itr = 100):
    
    error = np.zeros(itr)
    m = np.shape(A)[0]
    n_dim =  np.shape(A)[1]
    
    n = np.shape(X)
    x = np.ravel(X)
    A,b = row_normalised_mx(A,b,n_dim)
    
    y = np.zeros(np.shape(x)[0])    
    converge = True
    k = 0
    
    gamma = gamma*n_dim/m
            
    while converge == True and k < itr:
        try:
            t = permutation(np.arange(m))
            for j in range(m): #Inner iteration for Kaczmarz updates
                y_old = y
                a = A[t[j],:]
                y = y + gamma*(b[t[j]] - a@y)*a/(np.linalg.norm(a)**2)
                
                if (j+1)%period == 0:
                    y = y_old + lamda*(y - y_old)
                    WW = torch.reshape(torch.tensor(y), n)
                    y = vectorize_tl(HOSVD_rank_app(WW,r))
                    
            error[k] = np.linalg.norm(vectorize_np(y)-x)/np.linalg.norm(x)
            k = k+1 
                
        except np.linalg.LinAlgError:
            print("Doesn't converge")
            y = np.zeros(np.shape(x)[0])-1
            error = np.zeros(itr)+np.inf
            converge = False
    return y,error
