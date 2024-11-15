# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:50:57 2024

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
from astroML.datasets import generate_mu_z
from sklearn.linear_model import LinearRegression

#%%-----------------------------------Generate SN model------------------------------------------ 

z_sample, mu_sample, dmu = generate_mu_z(100)

plt.figure(figsize=(20, 20))
plt.errorbar(z_sample, mu_sample, dmu, marker="o", markersize=8, ls="", c="darkcyan", label="SN data")
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlabel("Redshift", size=35)
plt.ylabel("$\mu$", size=35)

matrix = np.stack((np.ones(len(z_sample)), z_sample, mu_sample), axis=1)

lr = LinearRegression()
lr.fit(z_sample[:, np.newaxis], mu_sample[:, np.newaxis], sample_weight=dmu)

theta_0 = lr.intercept_
theta_1 = lr.coef_

model = lr.predict(z_sample[:, np.newaxis])

x_grid = np.arange(0, max(z_sample), 1.)
plt.plot(x_grid, lr.predict(x_grid[:, np.newaxis]), c="crimson", lw=5)


