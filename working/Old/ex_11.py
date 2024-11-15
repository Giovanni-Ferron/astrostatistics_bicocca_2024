# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:35:39 2024

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import loguniform
from scipy.stats import uniform
import corner
import dynesty
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot


def Burst(t, b, A, t_0, alpha):
    y = []
    
    for i in t:
        if i < t_0:
            y.append(b)
        
        else:
            y.append(b + A * np.exp(-alpha * (i - t_0)))
        
    return np.array(y)


def PriorTransform(u):
    x = np.array(u)
    
    x[0] = 50 * u[0] #b
    x[1] = 50 * u[1] #A
    x[2] = 100 * u[2] #t_0
    x[3] = loguniform.ppf(u[3], np.exp(-5), np.exp(5)) #alpha
    
    return x


def LogLikelihood(params):   
    b, A, t_0, alpha = params
    data = file
    
    L = np.prod((1 / np.sqrt(2 * np.pi * data[:,2]**2)) * np.exp(-(data[:,1] - Burst(data[:,0], b, A, t_0, alpha))**2 / (2 * data[:,2]**2)))
    # print(L)
        
    return np.log(L)


file = np.load("../solutions/transient.npy")
time = file[:,0]
flux = file[:,1]

plt.figure(figsize=(20, 20))
plt.errorbar(time, flux, file[:,2], color="darkcyan")
plt.xticks(size=25)
plt.yticks(size=25)
plt.xlabel("Time", size=30)
plt.ylabel("Flux", size=30)

#%%------------------------------------Nested sampling-------------------------------------

ndim = 4

sampler = dynesty.NestedSampler(LogLikelihood, PriorTransform, ndim)
sampler.run_nested()
results = sampler.results

#%%

#%%-------------------------------------Plot results-----------------------------------------

plt.figure(figsize=(20, 20))
rfig, raxes = dyplot.runplot(results)
tfig, taxes = dyplot.traceplot(results)
cfig, caxes = dyplot.cornerplot(results)

#%%

#%%-------------------------------------Second model---------------------------------------

# def SecondBurst(t, b, A, t_0, s_w):
#     return b + A * np.exp(-(t - t_0)**2 / (2 * s_w**2))


# def PriorTransformSecond(u):
#     x = np.array(u)
    
#     x[0] = 50 * u[0] #b
#     x[1] = 50 * u[1] #A
#     x[2] = 100 * u[2] #t_0
#     x[3] = loguniform.ppf(u[3], np.exp(0), np.exp(2)) #s_w
    
#     return x


# def LogLikelihoodSecond(params):   
#     b, A, t_0, alpha = params
#     data = file
    
#     L = np.prod((1 / np.sqrt(2 * np.pi * data[:,2]**2)) * np.exp(-(data[:,1] - SecondBurst(data[:,0], b, A, t_0, alpha))**2 / (2 * data[:,2]**2)))
#     # print(L)
        
#     return np.log(L)


# sampler = dynesty.NestedSampler(LogLikelihood, PriorTransformSecond, 4)
# sampler.run_nested()
# results = sampler.results

# plt.figure(figsize=(20, 20))
# rfig, raxes = dyplot.runplot(results)
# tfig, taxes = dyplot.traceplot(results)
# cfig, caxes = dyplot.cornerplot(results)

#%%
