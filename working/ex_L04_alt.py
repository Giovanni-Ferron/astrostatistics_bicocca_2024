# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:50:59 2024

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
from astroML.datasets import fetch_dr7_quasar
from scipy.stats import rv_histogram
from scipy.optimize import brentq
from astropy.cosmology.realizations import Planck18
from astropy.cosmology import z_at_value

data = fetch_dr7_quasar()
data = data[:10000]

z = data["redshift"]

hist = np.histogram(z, bins=int(np.sqrt(len(z))))
hist_dist = rv_histogram(hist, density=True)

#Rejection sampling

N = 10000

x = np.random.uniform(min(z), max(z), N)
y = np.random.uniform(min(hist_dist.pdf(z)), max(hist_dist.pdf(z)), N)

r_samples = x[y < hist_dist.pdf(x)]

#Inverse transform sampling

unif = np.random.uniform(0, 1, N)

#Find the roots of cdf(x) - y = 0, where y is uniformly distributed

i_samples = [brentq(lambda p: hist_dist.cdf(p) - u, min(z), max(z)) for u in unif]

#%%
#Theoretical distribution
planck = Planck18

plt.figure(figsize=(15, 15))
plt.hist(z, bins=int(np.sqrt(len(z))), density=True, color="lightgrey", label="Redshift distribution")
plt.hist(r_samples, bins=int(np.sqrt(len(x))), density=True, histtype="step", lw=4, color="darkcyan", label="Rejection sampling")
plt.hist(i_samples, bins=int(np.sqrt(len(x))), density=True, histtype="step", lw=4, color="crimson", label="Inverse transform sampling")
# plt.hist(planck, bins=int(np.sqrt(len(x))), density=True, histtype="step", lw=4, color="crimson", label="Inverse transform sampling")
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlabel("z", size=30)
plt.ylabel("PDF", size=30)
plt.legend(prop={"size": 30})
#%%