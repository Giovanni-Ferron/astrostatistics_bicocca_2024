# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 23:40:08 2024

@author: Utente
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

#Set global plot parameters
plt.rcParams["figure.figsize"] = (30, 30)
plt.rcParams["font.size"] = 40
plt.rcParams["axes.titlesize"] = 50
plt.rcParams["axes.labelsize"] = 55

N = 10

#%%----------Generate data----------

mu = 1
sigma = 0.2
sample = norm(mu, sigma).rvs(N)

x_grid = np.linspace(0, 2.5, 1000)
L = np.prod([norm(i, sigma).pdf(x_grid) for i in sample], 0)

plt.figure()
plt.title("Quasar position measurments")
plt.xlabel("Position")

for i in sample:
    plt.plot(x_grid, norm(i, sigma).pdf(x_grid), linewidth=2)

#%%


#%%----------Maximum likelihood position----------

mu_max = x_grid[np.where(L == max(L))[0][0]]
mu_exp = np.mean(sample)

#%%


#%%----------Fisher matrix----------

F = np.diff(np.diff(np.log(L))) / np.diff(x_grid)[0]**2
sigma_F = np.sqrt(-(F[np.where(L == max(L))[0][0]])**-1)
sigma_mu = sigma / np.sqrt(N)

print("Maximum likelihood position: " + f"{mu_max: .3f}" + " +-" + f"{sigma_F: .3f}")
print("Sample mean position: " + f"{mu_exp: .3f}" + " +-" f"{sigma_mu: .3f}")

plt.plot(x_grid, L, lw=8, c="crimson", label="Analitycal likelihood")
plt.plot(x_grid, norm(mu_max, sigma_F).pdf(x_grid), lw=8, c="orange", label="Empirical likelihood")
plt.legend(prop={"size": 50})

#%%


#%%----------Heteroscedastic errors----------

sigma_i = norm(0.2, 0.05).rvs(N)
L_het = np.prod([norm(sample[i], sigma_i[i]).pdf(x_grid) for i in range(0, N)], 0)

mu_max_het = x_grid[np.where(L_het == max(L_het))[0][0]]
mu_exp_het = np.sum(sample / sigma_i**2) / np.sum(1 / sigma_i**2)

F_het = np.diff(np.diff(np.log(L_het))) / np.diff(x_grid)[0]**2
sigma_F_het = np.sqrt(-(F_het[np.where(L_het == max(L_het))[0][0]])**-1)
sigma_mu_het = np.sum(1 / sigma_i**2)**(-1/2)

print("Maximum likelihood position (heteroscedastic): " + f"{mu_max_het: .3f}" + " +-" + f"{sigma_F_het: .3f}")
print("Sample mean position (heteroscedastic) " + f"{mu_exp_het: .3f}" + " +-" f"{sigma_mu_het: .3f}")

#%%
