# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:54:54 2024

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
from astroML.datasets import generate_mu_z
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

#%%-----------------------------------Generate SN model------------------------------------------ 

z_sample, mu_sample, dmu = generate_mu_z(100)

plt.figure(figsize=(20, 20))
plt.errorbar(z_sample, mu_sample, dmu, marker="o", markersize=8, ls="", c="darkcyan", label="SN data")
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlabel("Redshift", size=35)
plt.ylabel("$\mu$", size=35)

#%%

#%%--------------------------------Gaussian Process Regerssion-----------------------------------------

flat_kernel = kernels.DotProduct()
gpr = GaussianProcessRegressor(kernel=flat_kernel)
gpr.fit(z_sample[:, np.newaxis], mu_sample)

mean, std = gpr.predict(z_sample[:, np.newaxis], return_std=True)

plt.errorbar(z_sample, mean, std, color="crimson")

#%%
