# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:08:32 2024

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt

#Generate random numbers and plot the distributions

N = 10000
x = np.random.uniform(0.1, 10, N)
y = np.log10(x)
h = abs(np.log(10) * 10**y) / (10 - 0.1)

plt.figure(figsize=(20, 20))
# plt.hist(x, bins=int(np.sqrt(10000)), histtype='step', lw=2, density=True)
plt.hist(y, bins=int(np.sqrt(N)), histtype='step', lw=2, density=True, label="$y = log(x)$")
plt.plot(y, h, lw=0, marker=".", markersize=10, c="crimson", label=r"$h(y) = |ln(10) \times 10^y|\ /\ 9.9$")
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlabel("y", size=30)
plt.ylabel("h(y)", size=30)
plt.legend(prop={"size": 40})

#Mean and median computation

print("x mean = " + str(np.log10(np.mean(x))))
print("y mean = " + str(np.mean(y)))

print("x median = " + str(np.log10(np.median(x))))
print("y median = " + str(np.median(y)))