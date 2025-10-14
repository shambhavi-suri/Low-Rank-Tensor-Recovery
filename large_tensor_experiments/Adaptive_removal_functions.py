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

## Projection onto the future ###
def proj_omega(y,x_1,x_2):
    ## Gram-Schmidt to find projection
    e_1 = x_1/np.linalg.norm(x_1)
    e_2 = x_2 - (x_2.T@e_1)*e_1
    e_2 = e_2/np.linalg.norm(e_2)
    
    y_proj_1 = (y.T@e_1)*e_1
    y_proj_2 = (y.T@e_2)*e_2
    
    return y_proj_1 + y_proj_2

def proj_vals(A,b,x_it,itr = 250,gamma = 1):
    
    error = np.zeros(itr)
    m = np.shape(A)[0]
    
    proj = []
    
    for j in range(m):
        a = A[j,:]
        proj_val = (b[j] - a@x_it)
        proj.append(proj_val)
    
    return proj


## Finding indices of high absolute residuals
def high_proj(a, number = 5):
    a = np.abs(a)
    res = sorted(range(len(a)), key = lambda sub: a[sub])[-number:]
    return res

def TIHT_HOSVD_proj_high(AA,yy,X,r,lamda = 1, itr = 100, num = 4): 
    
    n = np.shape(X)
    X_ravel = np.ravel(X)
    
    error = np.zeros(itr)
    
    vXX = torch.randn(n)*0

    for j in range(itr):
        
        proj = proj_vals(AA,yy,np.array(vectorize_tl(vXX)))
                           
        remove_row = high_proj(proj, number = num)
        A_1 = np.delete(AA,remove_row,axis = 0)
        y_1 = np.delete(yy,remove_row,axis = 0)
    
        WW = np.array(vectorize_tl(vXX)) + (m/(m-num))*lamda* np.matmul(A_1.T, (y_1 - np.matmul(A_1, np.array(vectorize_tl(vXX)))))
        WW = torch.reshape(torch.tensor(WW), n)
        vXX = HOSVD_rank_app(WW,r)
        error[j] = np.linalg.norm(vectorize_tl(vXX)- X_ravel)/np.linalg.norm(X_ravel)
          
    return vXX, error

def adaptive_TIHT_comp(AA,yy,X,r,lamda,mu = 1,itr=250, num_high = 5):
    
    n = np.shape(X)
    X_ravel = np.ravel(X)
    m = np.shape(AA)[0]
    
    error = np.zeros(itr)
    error_unclip = np.zeros(itr)
    delta_1_l = np.zeros(itr)
    rho_1_l = np.zeros(itr)
    rate_1_l = np.zeros(itr)
    delta_2_l = np.zeros(itr)
    rho_2_l = np.zeros(itr)
    rate_2_l = np.zeros(itr)
    rate_1_init = 1/np.linalg.norm(X_ravel)
    rate_2_init = 1/np.linalg.norm(X_ravel)
    
    vXX = torch.randn(n)*0
    vXX_unclip = torch.randn(n)*0

    for j in range(itr):
        
        delta_1 = lamda*(np.linalg.norm(np.matmul(AA, vectorize_tl(vXX_unclip) - X_ravel))/np.linalg.norm(vectorize_tl(vXX_unclip) - X_ravel))**2
        WW_unclip = np.array(vectorize_tl(vXX_unclip)) + mu*lamda* np.matmul(AA.T, (yy - np.matmul(AA, np.array(vectorize_tl(vXX_unclip)))))
        WW_unclip = torch.reshape(torch.tensor(WW_unclip), n)
        R_t = vectorize_tl(vXX_unclip) - X_ravel
        vXX_unclip = HOSVD_rank_app(WW_unclip,r)
        R_t1 = vectorize_tl(vXX_unclip) - X_ravel
        rho_1 = (np.linalg.norm(proj_omega((1/np.sqrt(m)*AA.T)@(1/np.sqrt(m)*AA)@R_t,R_t,R_t1))/np.linalg.norm((1/np.sqrt(m)*AA)@R_t))**2
        rate_1 = 2*np.sqrt(1-(2-mu*rho_1)*mu*delta_1)
        delta_1_l[j] = delta_1
        rho_1_l[j] = rho_1
        rate_1_l[j] = rate_1*rate_1_init
        rate_1_init = rate_1_l[j]
        error_unclip[j] = np.linalg.norm(vectorize_tl(vXX_unclip)- X_ravel)/np.linalg.norm(X_ravel)

        #With Clipping
        proj = proj_vals(AA,yy,np.array(vectorize_tl(vXX)))
        delta = AA@vectorize_tl(vXX) - yy


        remove_row = high_proj(proj, number = num_high)
        A_1 = np.delete(AA,remove_row,axis = 0)
        y_1 = np.delete(yy,remove_row,axis = 0)
        num = num_high
        
        delta_2 = (m/(m-num))*lamda*(np.linalg.norm(np.matmul(A_1, vectorize_tl(vXX) - X_ravel))/np.linalg.norm(vectorize_tl(vXX) - X_ravel))**2
        WW = np.array(vectorize_tl(vXX)) + (m/(m-num))*lamda*mu*np.matmul(A_1.T, (y_1 - np.matmul(A_1, np.array(vectorize_tl(vXX)))))
        WW = torch.reshape(torch.tensor(WW), n)
        R_t = vectorize_tl(vXX) - X_ravel
        vXX = HOSVD_rank_app(WW,r)
        R_t1 = vectorize_tl(vXX) - X_ravel
        rho_2 = (m/(m-num))*lamda*(np.linalg.norm(proj_omega(A_1.T@A_1@R_t,R_t,R_t1))/np.linalg.norm(A_1@R_t))**2
        rate_2 =  2*np.sqrt(1-(2-mu*rho_2)*mu*delta_2)
        delta_2_l[j] = delta_2
        rho_2_l[j] = rho_2
        rate_2_l[j] = rate_2*rate_2_init
        rate_2_init = rate_2_l[j]
        error[j] = np.linalg.norm(vectorize_tl(vXX)- X_ravel)/np.linalg.norm(X_ravel)
        
    return delta_1_l, delta_2_l, rho_1_l, rho_2_l, rate_1_l, rate_2_l, error_unclip, error

def adaptive_TIHT(AA,yy,X,r,lamda,mu = 1,itr=250, numb_high=5):
    
    n = np.shape(X)
    X_ravel = np.ravel(X)
    
    error = np.zeros(itr)
    vXX = torch.randn(n)*0
    m = np.shape(AA)[0]
    
    for j in range(itr):
        proj = proj_vals(AA,yy,np.array(vectorize_tl(vXX)))
        delta = AA@vectorize_tl(vXX) - yy
        
        remove_row = high_proj(proj, number = numb_high)
        A_1 = np.delete(AA,remove_row,axis = 0)
        num = numb_high
        
        WW = np.array(vectorize_tl(vXX)) + (m/(m-num))*lamda*mu*A_1.T@np.delete(proj,remove_row,axis = 0)
        WW = torch.reshape(torch.tensor(WW), n)
        vXX = HOSVD_rank_app(WW,r)
        error[j] = np.linalg.norm(vectorize_tl(vXX)- X_ravel)/np.linalg.norm(X_ravel)
    
    return vXX, error

def adaptive_KZTIHT(AA,yy,X,r,gamma = 1,lamda = 1,itr=250, numb_high=5):
    
    n = np.shape(X)
    X_ravel = np.ravel(X)
    n_dim =  np.shape(AA)[1]
    
    AA,yy = row_normalised_mx(AA,yy,n_dim)
    
    error = np.zeros(itr)
    y = torch.randn(n)*0
    m = np.shape(AA)[0]
    converge = True
    k = 0
    
    while converge == True and k < itr:
        proj = proj_vals(AA,yy,np.array(vectorize_tl(y)))
        
        remove_row = high_proj(proj, number = numb_high)
        A_1 = np.delete(AA,remove_row,axis = 0)
        y_1 = np.delete(yy,remove_row,axis = 0)
        num = numb_high
        
        gamma = gamma*n_dim/(m-num)
        y_old = y
        
        try:
            t = permutation(np.arange(m-num))
            for j in range(m-num): #Inner iteration for Kaczmarz updates
                a = A_1[t[j],:]
                y = y + gamma*(y_1[t[j]] - a@y)*a/(np.linalg.norm(a)**2)    
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
            
    
    return y, error

def adaptive_TIHT_CP(AA,yy,X,r,lamda,mu = 1,itr=250, numb_high=5):
    
    n = np.shape(X)
    X_ravel = np.ravel(X)
    
    error = np.zeros(itr)
    vXX = torch.zeros(n)
    vXX= np.array((vectorize_tl(vXX)))
    m = np.shape(AA)[0]
    
    for j in range(itr):
        proj = proj_vals(AA,yy,np.ravel(vXX))
        delta = AA@np.ravel(vXX) - yy
        
        remove_row = high_proj(proj, number = numb_high)
        A_1 = np.delete(AA,remove_row,axis = 0)
        y_1 = np.delete(yy,remove_row,axis = 0)
        A_1 = np.delete(AA,remove_row,axis = 0)
        num = numb_high

        WW = np.array(vectorize_tl(vXX)) + (m/(m-num))*A_1.T@lamda*mu*np.delete(proj,remove_row,axis = 0)
        WW = torch.reshape(torch.tensor(WW), n)
        vXX = CP_rank_app(WW,r)
        error[j] = np.linalg.norm(vectorize_np(vXX)-X_ravel)/np.linalg.norm(X_ravel)
    
def adaptive_KZTIHT_CP(A,b,X,r,gamma = 1,lamda = 1,itr=250, numb_high=5):
    
    error = np.zeros(itr)
    m = np.shape(A)[0]
    n_dim =  np.shape(A)[1]
    
    n = np.shape(X)
    x = np.ravel(X)
    A,b = row_normalised_mx(A,b,n_dim)
    
    y = np.zeros(np.shape(x)[0])    
    converge = True
    k = 0
    m_1 = m - numb_high

    
    gamma = gamma*n_dim/m_1
            
    while converge == True and k < itr:
        proj = proj_vals(A,b,np.ravel(y))
        remove_row = high_proj(proj, number = numb_high)
        
        A_1 = np.delete(A,remove_row,axis = 0)
        b_1 = np.delete(b,remove_row,axis = 0)
        
        try:
            y_old = y
            t = permutation(np.arange(m_1))
            for j in range(m_1): #Inner iteration for Kaczmarz updates
                a = A_1[t[j],:]
                y = y + gamma*(b_1[t[j]] - a@y)*a/(np.linalg.norm(a)**2)     
                
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
            
    
    return y, error

def adaptive_TIHT_comp(AA,yy,X,r,lamda,mu = 1,itr=250, num_high = 5):
    
    n = np.shape(X)
    X_ravel = np.ravel(X)
    m = np.shape(AA)[0]
    
    error = np.zeros(itr)
    error_unclip = np.zeros(itr)
    delta_1_l = np.zeros(itr)
    rho_1_l = np.zeros(itr)
    rate_1_l = np.zeros(itr)
    delta_2_l = np.zeros(itr)
    rho_2_l = np.zeros(itr)
    rate_2_l = np.zeros(itr)
    rate_1_init = 1/np.linalg.norm(X_ravel)
    rate_2_init = 1/np.linalg.norm(X_ravel)
    
    vXX = torch.randn(n)*0
    vXX_unclip = torch.randn(n)*0

    for j in range(itr):
        
        delta_1 = lamda*(np.linalg.norm(np.matmul(AA, vectorize_tl(vXX_unclip) - X_ravel))/np.linalg.norm(vectorize_tl(vXX_unclip) - X_ravel))**2
        WW_unclip = np.array(vectorize_tl(vXX_unclip)) + mu*lamda* np.matmul(AA.T, (yy - np.matmul(AA, np.array(vectorize_tl(vXX_unclip)))))
        WW_unclip = torch.reshape(torch.tensor(WW_unclip), n)
        R_t = vectorize_tl(vXX_unclip) - X_ravel
        vXX_unclip = HOSVD_rank_app(WW_unclip,r)
        R_t1 = vectorize_tl(vXX_unclip) - X_ravel
        rho_1 = (np.linalg.norm(proj_omega((1/np.sqrt(m)*AA.T)@(1/np.sqrt(m)*AA)@R_t,R_t,R_t1))/np.linalg.norm((1/np.sqrt(m)*AA)@R_t))**2
        rate_1 = 2*np.sqrt(1-(2-mu*rho_1)*mu*delta_1)
        delta_1_l[j] = delta_1
        rho_1_l[j] = rho_1
        rate_1_l[j] = rate_1*rate_1_init
        rate_1_init = rate_1_l[j]
        error_unclip[j] = np.linalg.norm(vectorize_tl(vXX_unclip)- X_ravel)/np.linalg.norm(X_ravel)

        #With Clipping
        proj = proj_vals(AA,yy,np.array(vectorize_tl(vXX)))
        delta = AA@vectorize_tl(vXX) - yy


        remove_row = high_proj(proj, number = num_high)
        A_1 = np.delete(AA,remove_row,axis = 0)
        y_1 = np.delete(yy,remove_row,axis = 0)
        num = num_high
        
        delta_2 = (m/(m-num))*lamda*(np.linalg.norm(np.matmul(A_1, vectorize_tl(vXX) - X_ravel))/np.linalg.norm(vectorize_tl(vXX) - X_ravel))**2
        WW = np.array(vectorize_tl(vXX)) + (m/(m-num))*lamda*mu*np.matmul(A_1.T, (y_1 - np.matmul(A_1, np.array(vectorize_tl(vXX)))))
        WW = torch.reshape(torch.tensor(WW), n)
        R_t = vectorize_tl(vXX) - X_ravel
        vXX = HOSVD_rank_app(WW,r)
        R_t1 = vectorize_tl(vXX) - X_ravel
        rho_2 = (m/(m-num))*lamda*(np.linalg.norm(proj_omega(A_1.T@A_1@R_t,R_t,R_t1))/np.linalg.norm(A_1@R_t))**2
        rate_2 =  2*np.sqrt(1-(2-mu*rho_2)*mu*delta_2)
        delta_2_l[j] = delta_2
        rho_2_l[j] = rho_2
        rate_2_l[j] = rate_2*rate_2_init
        rate_2_init = rate_2_l[j]
        error[j] = np.linalg.norm(vectorize_tl(vXX)- X_ravel)/np.linalg.norm(X_ravel)
        
    return delta_1_l, delta_2_l, rho_1_l, rho_2_l, rate_1_l, rate_2_l, error_unclip, error
