# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:33:55 2024

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

deaths = np.array([0, 1, 2, 3, 4])
groups = np.array([109, 65, 22, 3, 1])

prob = groups / 200
mu = np.average(deaths, weights=groups)

dist = poisson(mu)

plt.figure(figsize=(20, 20))
plt.bar(deaths, prob, color="springgreen", edgecolor="seagreen", label="Data")
plt.plot(deaths, dist.pmf(deaths), color="fuchsia", linewidth=8, label="Poisson")
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlabel("Deaths", size=30)
plt.ylabel("PDF", size=30)
plt.legend(prop={"size": 40})


