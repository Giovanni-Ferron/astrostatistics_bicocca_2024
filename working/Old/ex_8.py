# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:48:06 2024

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.neighbors import KernelDensity

#%%---------------------------Sample generation and histogram------------------------------

N = 10000
spin = np.random.uniform(0, 1, N)
M = np.random.normal(loc=1., scale=0.02, size=N)

f = np.sqrt((1 + np.sqrt(1 - spin**2)) / 2)
M_irr = M * f

#Histogram of M_irr with different bin sizes

plt.figure(figsize=(20, 20))
plt.hist(M_irr, bins="scott", color="darkcyan", alpha=0.7, density=True)
plt.hist(M_irr, bins="fd", color="darkorange", alpha=0.8, density=True)
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlabel("$M_{irr}$", size=30)
plt.ylabel("PDF", size=30)

#Apply Kernel Density Evaluation to the histogram

M_irr = np.sort(M_irr)
f = np.sort(f)

def KDE(kernel):
    kde = KernelDensity(bandwidth=0.01, kernel=kernel)
    kde.fit(M_irr[:, np.newaxis])
    
    return np.exp(kde.score_samples(M_irr[:, np.newaxis]))
    
plt.plot(M_irr, KDE("linear"), lw=6, color="green")
plt.plot(M_irr, KDE("tophat"), lw=6, color="blue")
plt.plot(M_irr, KDE("gaussian"), lw=6, color="red")

#%%

#%%---------------------------------------KS tests-----------------------------------------

#Compute and plot the KS distance of M_irr and f, and M_irr and M for various sigmas

def KSTest(sample_1, sample_2):
    cdf_1 = np.sort(sample_1) / len(sample_1)
    cdf_2 = np.sort(sample_2) / len(sample_2)
    
    return ks_2samp(cdf_1, cdf_2)


ks_Mf = []
ks_MM = []
sigmas_low = np.arange(0.0001, 0.005, 0.0001)
sigmas_high = np.arange(100., 500., 1.)

for i in sigmas_low:
    M_i = np.random.normal(loc=1, scale=i, size=N)
    ks_Mf.append(KSTest(M_i * f, f)[1])
    ks_MM.append(KSTest(M_i * f, M_i)[1])
    
for i in sigmas_high:
    M_i = np.random.normal(loc=1, scale=i, size=N)
    ks_Mf.append(KSTest(M_i * f, f)[1])
    ks_MM.append(KSTest(M_i * f, M_i)[1])

#Low sigma limit

plt.figure(figsize=(20, 20))
plt.plot(sigmas_low, ks_Mf[:len(sigmas_low)], lw=4, color="darkcyan")
plt.plot(sigmas_low, ks_MM[:len(sigmas_low)], lw=4, color="darkorange")
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlabel("$\sigma$", size=30)
plt.ylabel("KS p value", size=30)

#High sigma limit

plt.figure(figsize=(20, 20))
plt.plot(sigmas_high, ks_Mf[len(sigmas_low):], lw=4, color="darkcyan")
plt.plot(sigmas_high, ks_MM[len(sigmas_low):], lw=4, color="darkorange")
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlabel("$\sigma$", size=30)
plt.ylabel("KS p value", size=30)

#%%
