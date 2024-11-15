# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:46:06 2024

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

#%%--------------------------------------Part 1------------------------------------------
N = 10
sigma = 0.2
data = np.random.normal(loc=1.0, scale=sigma, size=N)

x = np.arange(0., 2.5, 0.01)

#Likelihood computation and plotting
plt.figure(figsize=(20, 20))
plt.xticks(size=30)
plt.yticks(size=30)

L = st.norm(loc=data[0], scale=0.2).pdf(x)

for i in range(1, N):
    if i != 0:
        L *= st.norm(loc=data[i], scale=0.2).pdf(x)
        
    plt.plot(x, st.norm(loc=data[i], scale=0.2).pdf(x))
    
plt.plot(x, L, lw=4)

#Get the maximum likelihood

L_max = x[np.argsort(L)[-1]]

print("\nHomoscedastic errors:")
print("Likelihood maximum point = " + f"{L_max: .3f}")

#Compute the MLE for the mean and its unceartainty

mean = sum(data) / N
std = sigma / np.sqrt(N)

print("MLE =" + f"{mean: .3f}" + " +-" + f"{std: .3f}")
#%%

#%%--------------------------------------Part 2------------------------------------------
logL = np.log(L)

dLdx_2 = np.diff(np.diff(logL))
F = np.sqrt(-dLdx_2 / 0.01**2)**-1

plt.plot(x, st.norm(loc=L_max, scale=F[np.argsort(L)[-1]]).pdf(x), lw=4)

print("Fisher matrix error =" + f"{F[0]: .3f}")

#%%

#%%--------------------------------------Part 3----------------------------------------
#Generate heteroscedastic errors

sigma_i = np.random.normal(loc=0.2, scale=0.05, size=N)

#Likelihood computation and plotting

plt.figure(figsize=(20, 20))
plt.xticks(size=30)
plt.yticks(size=30)

L_het = st.norm(loc=data[0], scale=sigma_i[0]).pdf(x)

for i in range(1, N):
    if i != 0:
        L_het *= st.norm(loc=data[i], scale=sigma_i[i]).pdf(x)
        
    plt.plot(x, st.norm(loc=data[i], scale=sigma_i[i]).pdf(x))
    
plt.plot(x, L_het, lw=4)

#Get the maximum likelihood

L_het_max = x[np.argsort(L_het)[-1]]

print("\nHeteroscedastic errors:")
print("Likelihood maximum point =" + f"{L_het_max: .3f}")

#Compute the MLE for the mean and its unceartainty

mean_het = sum(data / sigma_i**2) / sum(1/sigma_i**2)
std_het = (sum(1/sigma_i**2))**(-1/2)

print("MLE =" + f"{mean_het: .3f}" + " +-" + f"{std_het: .3f}")

#Fisher matrix error

logL_het = np.log(L_het)

dLdx_2_het = np.diff(np.diff(logL_het))
F_het = np.sqrt(-dLdx_2_het / 0.01**2)**-1

plt.plot(x, st.norm(loc=L_het_max, scale=F_het[np.argsort(L_het)[-1]]).pdf(x), lw=4)

print("Fisher matrix error =" + f"{F_het[0]: .3f}")


#%%
