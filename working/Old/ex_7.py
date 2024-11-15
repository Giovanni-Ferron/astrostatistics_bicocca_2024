# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:46:58 2024

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

data = np.load("../../solutions/formationchannels.npy")

N = 10
gm = []
AIC = []

#Train the gaussian mixture model and compute the AIC

for i in range(1, N+1):
    gm.append(GaussianMixture(n_components=i).fit(data))
    AIC.append(gm[i-1].aic(data))
    
best = np.where(AIC == min(AIC))[0][0]
print("The best fitting model is " + str(best + 1) + " gaussians")

plt.figure(figsize=(20,20))
plt.xticks(np.arange(0, 10), size=30)
plt.yticks(size=30)
plt.plot(np.arange(1, N+1), AIC, lw=3, color="darkcyan", marker="o", markersize=10, label="AIC of the models")
plt.xlabel("N gaussians", size=30)
plt.ylabel("AIC", size=30)
plt.legend(prop={"size": 30})

#Compute and plot the sample likelihood

logL = gm[best].score_samples(data)

plt.figure(figsize=(20,20))
plt.xticks(size=30)
plt.yticks(size=30)
plt.hist(data, 300, density=True, label="Sample dataset", histtype="step", color="teal", lw=2.5)
plt.scatter(data, np.exp(logL), color="red", label="Likelihood of the sample")
plt.xlabel("Dataset", size=30)
plt.ylabel("PDF", size=30)
plt.legend(prop={"size": 30})

#Probability of finding the samples in a certain gaussian

gauss_1 = gm[best].predict_proba(data)[:,0]
gauss_2 = gm[best].predict_proba(data)[:,1]
gauss_3 = gm[best].predict_proba(data)[:,2]

plt.figure(figsize=(20,20))
plt.xticks(size=30)
plt.yticks(size=30)
plt.scatter(data[:,0], gauss_1, color="darkcyan", s=20, label="Gaussian 1")
plt.scatter(data[:,0], gauss_2, color="orange", s=20, label="Gaussian 2")
plt.scatter(data[:,0], gauss_3, color="forestgreen", s=20, label="Gaussian 3")
plt.xlabel("Dataset", size=30)
plt.ylabel("Probability", size=30)
plt.legend(prop={"size": 30})

    