# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 09:43:45 2024

@author: Utente
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def p(sigma, N):
    gaus = np.random.normal(loc=0., scale=sigma, size=N)
    return gaus[gaus > 0]

def f(x):
    return x**3

N = np.arange(10, 10010, 10, "int16")
integral_mc = []

for i in range(0, len(N)):
    sigma = 10
    integral_mc.append(sigma * np.sqrt(2 * np.pi) * np.mean(f(p(sigma, N[i]))) / 2)
    integral = 2 * sigma**4
    # print("Integral with Monte Carlo = " + str(integral_mc) + " +- " + str(sigma * np.sqrt(2 * np.pi) * np.std(f(p(sigma, N))) / np.sqrt(N)))

# print("Integral = " + str(integral))

plt.figure(figsize=(20, 20))
plt.plot(N, integral_mc, c="mediumturquoise")
plt.plot(N, np.ones(len(N)) * integral, lw=10, c="crimson")

