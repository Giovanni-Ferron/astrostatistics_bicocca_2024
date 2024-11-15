# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 21:00:36 2024

@author: Utente
"""

import numpy as np
from matplotlib import pyplot as plt
from astroML.datasets import fetch_dr7_quasar
from scipy.interpolate import interp1d
from scipy.stats import ks_2samp
from scipy.integrate import quad
from astropy.cosmology import Planck18

#Set global plot parameters
plt.rcParams["figure.figsize"] = (30, 30)
plt.rcParams["font.size"] = 40
plt.rcParams["axes.titlesize"] = 50
plt.rcParams["axes.labelsize"] = 55

data = fetch_dr7_quasar()
z = data["redshift"][:10000]
N = len(z) 

#%%----------Data histogram----------

mu_z = np.mean(z)
sigma_z = np.std(z)
sigma_mu = sigma_z / np.sqrt(N)

fig = plt.figure()
ax = fig.add_subplot()
hist = plt.hist(z, density=True, bins="fd", color="darkcyan", histtype="step", linewidth=8)
# plt.plot(z, 0*z, "|", markersize=50, color="black")
plt.vlines(mu_z, 0, 1, transform=ax.get_xaxis_transform(), linestyles="-.", linewidth=8, color="crimson", label="$\mu_z = $" + f"{mu_z: .2f}" + "$\pm$" + f"{sigma_mu: .2f}")
plt.vlines(mu_z - sigma_z, 0, 1, transform=ax.get_xaxis_transform(), linestyles="--", linewidth=8, color="crimson", label="$\sigma_z = $" + f"{sigma_z: .2f}")
plt.vlines(mu_z + sigma_z, 0, 1, transform=ax.get_xaxis_transform(), linestyles="--", linewidth=8, color="crimson")
plt.fill_between([mu_z - sigma_z, mu_z + sigma_z], 1, color="crimson", alpha=0.06, transform=ax.get_xaxis_transform())
plt.title("Redshift distribution")
plt.xlabel("z")

#%%


#%%----------Rejection sampling----------

heights = hist[0]
bins = hist[1]
n_couples = 4 * N

unif_x = np.random.uniform(min(z), max(z), n_couples)
unif_y = np.random.uniform(0, max(heights), n_couples)

x_to_bin = np.digitize(unif_x, bins) - 1

sample = unif_x[np.where(unif_y < heights[x_to_bin])]


#%%


#%%----------Inverse transform sampling----------

#----------Empirical CDF----------

z = np.sort(z)
e_cdf = [len(np.where(z < i)[0]) / N for i in z]

fig_cdf = plt.figure()
ax_cdf = fig_cdf.add_subplot(1, 1, 1)
plt.plot(z, e_cdf, linewidth=6, color="darkcyan")
plt.title("Redshift empirical CDF")
plt.xlabel("z")
plt.grid()

#----------Interpolate inverse eCDF and sample the ePDF----------

i_cdf = interp1d(e_cdf, z)
unif_cdf = np.random.uniform(0, max(e_cdf), N)
z_pdf = i_cdf(unif_cdf)

#%%

#%%----------Plot the distributions----------

plt.figure()
plt.hist(z, density=True, bins="fd", color="darkcyan", histtype="step", linewidth=8, label="True distribution")
plt.hist(sample, density=True, bins="fd", color="green", histtype="step", linewidth=8, label="Rejection sampling")
plt.hist(z_pdf, density=True, bins="fd", color="crimson", histtype="step", linewidth=8, label="Inverse transform sampling")
plt.title("Redshift empirical CDF")
plt.xlabel("z")
plt.legend(prop={"size": 55})
plt.grid()

#----------Compare the distributions to the original one----------

print("Rejection sampling KS test:\n" + str(ks_2samp(z, sample)) + "\n")
print("Inverse transform sampling KS test:\n" + str(ks_2samp(z, z_pdf)))

#%%

#%%----------Distribution of quasars----------

def qso_pdf(z):
    return 4 * np.pi * Planck18.differential_comoving_volume(z).value

norm = quad(qso_pdf, 0, max(z))[0]

ax.plot(z, 2.7 * qso_pdf(z) / norm, lw=8, c="green", label="Predicted quasar distribution")
ax.legend(prop={"size": 50})

#%%