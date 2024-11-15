# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:52:59 2024

@author: Utente
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import ks_2samp

#Set global plot parameters
plt.rcParams["figure.figsize"] = (30, 30)
plt.rcParams["font.size"] = 40
plt.rcParams["axes.titlesize"] = 50
plt.rcParams["axes.labelsize"] = 55
# plt.rcdefaults()

def f(x):
    return np.sqrt((1 + np.sqrt(1 - x**2)) / 2)

N = 1000

#Spin
chi = np.random.uniform(0, 1, N)

#Mass
mu = 1
sigma = 0.02
M = np.random.normal(mu, sigma, N)

#Irreducible mass
M_irr = np.sort(M * f(chi))

#%%----------Irreducible mass distribution----------

fig = plt.figure()
ax = fig.add_subplot()
ax.hist(M_irr, color="darkcyan", alpha=0.5, bins="fd", density=True)
plt.title("Irreducible mass distribution")
plt.xlabel("$M_{irr}\ [\mu = 1]$")

#----------Kernel density estimation----------

kde = KernelDensity(bandwidth=0.01, kernel="gaussian")
kde.fit(M_irr[:, np.newaxis])
kde_logL = kde.score_samples(M_irr[:, np.newaxis])

ax.plot(M_irr, np.exp(kde_logL), color="crimson", lw=8)

#%%


#%%----------KS distance----------

sigma_grid = np.linspace(0.001, 10, 100)
D_Mf = []
D_MM = []

for i in sigma_grid:
    M_i = np.random.normal(mu, i, N)
    D_Mf.append(ks_2samp(f(chi), M_i * f(chi)).statistic)
    D_MM.append(ks_2samp(M_i, M_i * f(chi)).statistic)
    
D_Mf = np.array(D_Mf)
D_MM = np.array(D_MM)
    
plt.figure()
plt.plot(sigma_grid, D_Mf, c="darkcyan", lw=8, label="$M_{irr} - f$ distance")
plt.plot(sigma_grid, D_MM, c="crimson", lw=8, label="$M_{irr} - M$ distance")
plt.title("KS distance")
plt.grid()
plt.legend(prop={"size": 55})

#%%

#%%----------Analitycal pdf of M_irr----------

p_f = 2 * (2 * f(chi)**2 - 1) / (N * f(chi) * np.sqrt(1 - f(chi)**2)) 

def p_Mirr(M):
    s = np.random.uniform(1/np.sqrt(2), 1, N)
    pdf = np.array([(1 - 1/np.sqrt(2)) / N * np.sqrt(2/np.pi)/sigma * np.sum(np.exp(-(i/s - mu)**2/(2*sigma**2)) * (2*s**2 - 1)/(s * np.sqrt(1 - s**2))) for i in M])

    return pdf

ax.plot(M_irr, p_Mirr(M_irr), lw=8, color="green")

#%%
