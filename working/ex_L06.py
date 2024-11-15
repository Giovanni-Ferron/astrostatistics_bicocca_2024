# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:44:19 2024

@author: Utente
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

#Set global plot parameters
plt.rcParams["figure.figsize"] = (30, 30)
plt.rcParams["font.size"] = 40
plt.rcParams["axes.titlesize"] = 50
plt.rcParams["axes.labelsize"] = 55

data = np.load("../solutions/formationchannels.npy")
data[:,0] = np.sort(data[:,0])

#%%----------Plot data----------

fig_data = plt.figure()
ax = fig_data.add_subplot()
ax.hist(data, alpha=0.5, color="darkcyan", bins="fd", density=True)
plt.title("Black hole masses distribution")
plt.xlabel("$M_{BH}\ [M_\odot]$")

#%%


#%%----------Gaussian mixture model----------

gm = []
N_gauss = np.arange(1, 11)

for i in N_gauss:
    gm.append(GaussianMixture(n_components = i).fit(data))

AIC = [gm[i].aic(data) for i in N_gauss - 1]
best_model = N_gauss[np.argsort(AIC)[0]] - 1
gm_best = gm[best_model]

plt.figure()
plt.plot(N_gauss, AIC, color="crimson", lw=8)
plt.title("Gaussian mixture models")
plt.xlabel("$N_{gaussians}$")
plt.ylabel("AIC")
plt.grid()

#----------Plot best model over data----------

x_grid = np.linspace(0, 60, 1000)
models = np.array(gm_best.score_samples(x_grid[:, np.newaxis]))

ax.plot(x_grid, np.exp(models), c="crimson", lw=6)

#----------Responsibilities----------

gauss = np.array([gm_best.predict_proba(data)[:, i] for i in range(0, best_model + 1)])

plt.figure()
plt.plot(data, gauss[0], c="crimson", lw=6, label="Gaussian 1")
plt.plot(data, gauss[1], c="green", lw=6, label="Gaussian 2")
plt.plot(data, gauss[2], c="darkcyan", lw=6, label="Gaussian 3")
plt.title("Probability of the dataset")
plt.xlabel("$M_{BH}\ [M_\odot]$")
plt.ylabel("Probability density")
plt.grid()
plt.legend(prop={"size": 50})

#%%