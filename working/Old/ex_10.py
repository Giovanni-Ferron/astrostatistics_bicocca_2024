# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:42:21 2024

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee


def Burst(t, b, A, t_0, alpha):
    y = []
    
    for i in t:
        if i < t_0:
            y.append(b)
        
        else:
            y.append(b + A * np.exp(-alpha * (i - t_0)))
        
    return np.array(y)


def Prior_b(b, A, t_0, alpha):
    if b < 0 or b > 50:
        return 0.
    
    return 1/50

def Prior_A(b, A, t_0, alpha):
    if A < 0 or A > 50:
        return 0.
    
    return 1/50

def Prior_t_0(b, A, t_0, alpha):
    if t_0 < 0 or t_0 > 100:
        return 0.
    
    return 1/100

def Prior_alpha(b, A, t_0, alpha):
    if alpha < np.exp(-5) or alpha > np.exp(5):
        return 0.
    
    return 1 / (10 * alpha)


def Likelihood(data, b, A, t_0, alpha):
    return np.prod(np.exp(-(data[:,1] - Burst(data[:,0], b, A, t_0, alpha))**2 / (2 * data[:,2]**2)))


def LogPosterior(params, data):   
    b, A, t_0, alpha = params  
    
    L = Likelihood(data, b, A, t_0, alpha) * Prior_b(b, A, t_0, alpha) * Prior_A(b, A, t_0, alpha) * Prior_t_0(b, A, t_0, alpha) * Prior_alpha(b, A, t_0, alpha)
    print(L)
        
    return np.log(L)


data = np.load("../solutions/transient.npy")
time = data[:,0]
flux = data[:,1]

plt.figure(figsize=(20, 20))
plt.errorbar(data[:,0], data[:,1], data[:,2], color="darkcyan")
plt.xticks(size=25)
plt.yticks(size=25)
plt.xlabel("Time", size=30)
plt.ylabel("Flux", size=30)

#%%----------------------------------Generate the chain------------------------------------

ndim = 4
nwalkers = 20
nsteps = 40000

start = np.random.random((nwalkers, ndim)) + [min(flux) + 3, max(flux) - min(flux) + 1, time[np.where(flux == max(flux))[0][0]] + 1, 0.13]
sampler = emcee.EnsembleSampler(nwalkers, ndim, LogPosterior, args=[data])
sampler.run_mcmc(start, nsteps, progress=True)

#%%

#%%-----------------------------------Plot the chain-----------------------------------------

tau = sampler.get_autocorr_time()
thin = int(max(tau))
burn = int(2 * max(tau))

mc_trace = sampler.get_chain(discard=burn, thin=thin, flat=True)
chain = mc_trace.flatten()

b = mc_trace[:,0]
A = mc_trace[:,1]
t_0 = mc_trace[:,2]
alpha = mc_trace[:,3]

fig_1 = corner.corner(mc_trace, color="#017a79", quantiles=[0.68, 0.95], levels=[0.39, 0.68, 0.95], labels=["$b$", "$A$", "$t_0$", "$\\alpha$"], label_kwargs={"size": 15})
axes = np.array(fig_1.axes).reshape((ndim, ndim))

for y_i in range(0, ndim):
    for x_i in range(0, y_i):
        ax = axes[y_i, x_i]
        ax.tick_params(which="both", labelsize=10)
        ax.grid(color="gainsboro")
        ax.axvline(mc_trace[:,x_i].mean(), c="red")
        ax.axhline(mc_trace[:,y_i].mean(), c="red")
        ax.plot(mc_trace[:,x_i].mean(), mc_trace[:,y_i].mean(), c="red", marker="o")
        
        if y_i == ndim - 1:
            ax.set_ylim(0., 0.3)
            
for i in range(0, ndim):
    ax = axes[i, i]
    ax.tick_params(which="both", labelsize=10)
    ax.axvline(mc_trace[:,i].mean(), c="red")

#%%

#%%------------------------------------Posterior spread------------------------------------

samples = mc_trace[np.random.choice(mc_trace.shape[0], replace=False, size=100)]

plt.figure(figsize=(20, 20))
plt.plot(time, flux)

for i in samples:
    plt.plot(time, Burst(time, i[0], i[1], i[2], i[3]), alpha=0.5)
    
plt.xticks(size=25)
plt.yticks(size=25)
plt.xlabel("Time", size=30)
plt.ylabel("Flux", size=30)

#%%

#%%------------------------------------Parameter median------------------------------------

#Using the medians of the marginalized posteriors as best parameters (summary statistics) of the model is wrong because the vector of parameter medians doesn't exist in the 4D space
#We should instead use, for example, the parameters that maximize the posterior (maximum a posteriori), which corresponds to an actual set of parameters entries on the chain

fig_2 = plt.figure(figsize=(20, 20))

chain_params = [b, A, t_0, alpha]
params_name = ["b", "A", "t_0", "\\alpha"]
params_median = np.array([np.median(p) for p in chain_params])
params_quantiles = np.array([np.quantile(b, [0.05, 0.95]), corner.quantile(A, [0.05, 0.95]), corner.quantile(t_0, [0.05, 0.95]), corner.quantile(alpha, [0.05, 0.95])])
lower = params_median - params_quantiles[:,0]
higher = params_quantiles[:,1] - params_median

for i in range(1, len(chain_params) + 1):
    ax_2 = fig_2.add_subplot(2, 2, i)
    ax_2.hist(chain_params[i - 1], "fd", histtype="step", color="#017a79", lw=3, density=True)
    ax_2.axvline(params_median[i - 1], c="black", zorder=3, label="$" + params_name[i - 1] + "\ = " + f"{params_median[i - 1]: .2f}" + "^{+" + f"{higher[i - 1]: .2f}" + "}_{-" + f"{lower[i - 1]: .2f}" + "}$")
    ax_2.axvline(params_quantiles[i - 1][0], c="crimson", linestyle="dashed")
    ax_2.axvline(params_quantiles[i - 1][1], c="crimson", linestyle="dashed")
    ax_2.tick_params(which="both", labelsize=20)
    ax_2.set_xlabel("$" + params_name[i - 1] + "$", size=30)
    ax_2.fill_between(np.arange(params_quantiles[i - 1][0], params_quantiles[i - 1][1] + 0.0001, 0.0001), 0, 1, color="#fd5956", alpha=0.05, transform=ax_2.get_xaxis_transform())
    ax_2.legend(prop={"size": 20}, loc="upper right")

#%%
