# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:29:33 2024

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.cluster import KMeans, MeanShift
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

#%%------------------------------------Read data---------------------------------------------

req = requests.get("https://user-web.icecube.wisc.edu/~grbweb_public/Summary_table.txt")

with open("Summary_table.txt", "wb") as file:
    file.write(req.content)
    
data = np.loadtxt("Summary_table.txt", dtype="str", unpack=True)

with open("Summary_table.txt", "r") as f:
    names = np.array([n.strip().replace(" ", "_") for n in f.readlines()[1].replace("#","").replace("\n","").lstrip().split('    ') if n.strip()!=''])
    
#Dictionary with data

grb = dict(zip(names,data))

for lab in ['T90', "fluence", "fluence_error", "redshift", "T90_error"]:
    grb[lab] = np.array(grb[lab], dtype=float)
    
#%%

#%%-----------------------------------Assign data--------------------------------------------------

T90 = grb["T90"][~np.isnan(np.log(grb["T90"]))]
fluence = grb["fluence"][~np.isnan(np.log(grb["T90"]))]
redshift = grb["redshift"][~np.isnan(np.log(grb["T90"]))]

#%%

#%%--------------------------------Cross validation---------------------------------------------

def DoCV(estimator, param, train_data):
    cluster_num = np.arange(1, 4)
    
    grid = GridSearchCV(estimator, {param[0]: param[1]}, cv=5, verbose=3)
    grid.fit(train_data)
    
    return grid.best_params_[param[0]]
#%%--------------------------------K-means clustering---------------------------------------

def DoKMeans(train_data, n_clusters):  #Pass n_clusters as array if doCV is true
    train_data = train_data[:, np.newaxis] 
    
    clf = KMeans(n_clusters, n_init="auto")
    
    clf.fit(train_data)
    labels = clf.predict(train_data)
    centers = clf.cluster_centers_
        
    return labels, centers

#%%

#%%-------------------------------Mean-shift clustering---------------------------------------------

def DoMS(train_data, bw):
    train_data = train_data[:, np.newaxis] 
    scaler = preprocessing.StandardScaler()
    
    ms = MeanShift(bandwidth=bw, bin_seeding=True, cluster_all=True)
    ms.fit(scaler.fit_transform(train_data))
    
    labels = ms.labels_
    centers = ms.cluster_centers_
    
    n_clusters = len(np.unique(labels)[np.unique(labels) >= 0])
    print(n_clusters)
        
    return labels, centers

#%%

#%%-----------------------------Kernel Density Estimation----------------------------------------

def DoKDE(dataset, x, bw, kernel="linear"): #Pass bw as array if doCV is true
    dataset = dataset[:, np.newaxis]    
    
    kde = KernelDensity(bandwidth=bw, kernel=kernel)
    kde.fit(dataset)
    
    return kde.score_samples(np.log10(x[:, np.newaxis]))

#%%

#%%----------------------Clustering of T90 with different methods---------------------------------

#Kernel Density Estimation of the T90 histogram

x_grid = np.logspace(np.log10(min(T90)), np.log10(max(T90)), len(T90))
T90_KD = np.exp(DoKDE(np.log10(T90), x_grid, bw=0.1, kernel="gaussian"))

x_grid = np.logspace(np.log10(min(T90)), np.log10(max(T90)), len(T90))
T90_KD = np.exp(DoKDE(np.log10(T90), x_grid, bw=0.1, kernel="gaussian"))

#----------K-Means clustering----------

labels_K, centers_K = DoKMeans(np.log10(T90), n_clusters=2)
labels_K, centers_K = DoKMeans(np.log10(T90), n_clusters=2)
centers_K = 10**np.squeeze(centers_K)

#----------Mean-shift clustering----------

labels_ms, centers_ms = DoMS(np.log10(T90), 0.4)
centers_ms = 10**np.squeeze(centers_ms)

#----------Separation between clusters----------

cluster_lGRB = np.log10(T90)[labels_K == np.where(centers_K == max(centers_K))[0][0]]
cluster_sGRB = np.log10(T90)[labels_K == np.where(centers_K == min(centers_K))[0][0]]

treshold_lGRB = 10**min(cluster_lGRB)
treshold_sGRB = 10**max(cluster_sGRB)

#%%

#%%------------------Clustering of the redshift with different methods---------------------------------

#Kernel Density Estimation of the redshift histogram

z = grb["redshift"][~np.isnan(np.log(grb["redshift"]))]

x_grid_z = np.logspace(np.log10(min(z)), np.log10(max(z)), len(z))
z_KD = np.exp(DoKDE(np.log10(z), x_grid_z, bw=0.09, kernel="gaussian"))

#----------K-Means clustering----------

labels_K_z, centers_K_z = DoKMeans(np.log10(z), n_clusters=2)
centers_K_z = 10**np.squeeze(centers_K_z)

#----------Mean-shift clustering----------

labels_ms_z, centers_ms_z = DoMS(np.log10(z), 1.)
centers_ms_z = 10**np.squeeze(centers_ms_z)
print(centers_ms_z)

plt.figure(figsize=(20, 20))
plt.plot(x_grid_z, z_KD)
plt.axvline(centers_K_z[0], c="green", ls="--", lw=4, label="K-Means clustering")
plt.axvline(centers_K_z[1], c="green", ls="--", lw=4)
plt.axvline(centers_ms_z[0], c="red", ls="-.", lw=4, label="Mean-shift clustering")
plt.axvline(centers_ms_z[1], c="red", ls="-.", lw=4)
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlabel("Redshift", size=30)
plt.title("Redshift distribution", size=45)
plt.xscale("log")
plt.legend(prop={"size": 30})

#%%

#%%------------------Clustering of the fluence with different methods---------------------------------

#Kernel Density Estimation of the redshift histogram

f = grb["fluence"][np.logical_and(~np.isnan(np.log10(grb["fluence"])), np.log10(grb["fluence"]) != -np.inf)]

x_grid_f = np.logspace(np.log10(min(f[f != 0])), np.log10(max(f)), len(f))
f_KD = np.exp(DoKDE(np.log10(f), x_grid_f, bw=0.08, kernel="gaussian"))

#----------K-Means clustering----------

labels_K_f, centers_K_f = DoKMeans(np.log10(f), n_clusters=2)
centers_K_f = 10**np.squeeze(centers_K_f)

#----------Mean-shift clustering----------

labels_ms_f, centers_ms_f = DoMS(np.log10(f), 1.)
centers_ms_f = 10**np.squeeze(centers_ms_f)
print(centers_ms_f)

plt.figure(figsize=(20, 20))
plt.plot(x_grid_f, f_KD)
plt.axvline(centers_K_f[0], c="green", ls="--", lw=4, label="K-Means clustering")
plt.axvline(centers_K_f[1], c="green", ls="--", lw=4)
plt.axvline(centers_ms_f[0], c="red", ls="-.", lw=4, label="Mean-shift clustering")
plt.axvline(centers_ms_f[1], c="red", ls="-.", lw=4)
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlabel("Fluence $[erg\ cm{-2}]$", size=30)
plt.title("Fluence distribution", size=45)
plt.xscale("log")
plt.legend(prop={"size": 30})

#%%

#%%---------------------------------Plot data----------------------------------------------

#T90 histogram and clustering centers

plt.figure(figsize=(20, 20))
plt.plot(x_grid, T90_KD)
plt.axvline(centers_K[0], c="green", ls="--", lw=4, label="K-Means clustering")
plt.axvline(centers_K[1], c="green", ls="--", lw=4)
plt.axvline(centers_ms[0], c="red", ls="-.", lw=4, label="Mean-shift clustering")
plt.axvline(centers_ms[1], c="red", ls="-.", lw=4)
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlabel("$T_{90}\ [s]$", size=30)
plt.title("$T_{90}$ distribution", size=45)
plt.xscale("log") 
plt.legend(prop={"size": 30})
   
#Fluence vs T90 plot with labels

fig_f = plt.figure(figsize=(40, 20))
ax_f1 = fig_f.add_subplot(121)
ax_f2 = fig_f.add_subplot(122)

grb_type = np.zeros(2, dtype="U10")
grb_type[np.where(centers_K == max(centers_K))[0][0]] = "Long GRB"
grb_type[np.where(centers_K == min(centers_K))[0][0]] = "Short GRB"

#K-means labels

for i in np.unique(labels_K):
    ax_f1.scatter(T90[labels_K == i], fluence[labels_K == i], color="C" + str(i), alpha=0.3, label=grb_type[i]) 

ax_f1.tick_params(labelsize=30)
ax_f1.set_xlabel("$T_{90}\ [s]$", size=30)
ax_f1.set_ylabel("Fluence $[erg\ cm^{-2}]$", size=30)
ax_f1.set_title("K-Means clustering", size=40)
ax_f1.set_xscale("log")
ax_f1.set_yscale("log")
ax_f1.legend(prop={"size": 30})

#Mean-shift labels

for i in np.unique(labels_ms):
    ax_f2.scatter(T90[labels_ms == i], fluence[labels_ms == i], color="C" + str(i), alpha=0.3) 

ax_f2.tick_params(labelsize=30)
ax_f2.set_xlabel("$T_{90}\ [s]$", size=30)
ax_f2.set_title("Mean-shift clustering", size=40)
ax_f2.set_xscale("log")
ax_f2.set_yscale("log")


#Redshift vs T90 plot with labels

fig_z = plt.figure(figsize=(40, 20))
ax_z1 = fig_z.add_subplot(121)
ax_z2 = fig_z.add_subplot(122)

#K-means labels

for i in np.unique(labels_K):
    ax_z1.scatter(T90[labels_K == i], redshift[labels_K == i], color="C" + str(i), alpha=0.7, label=grb_type[i]) 

ax_z1.tick_params(labelsize=30)
ax_z1.set_xlabel("$T_{90}\ [s]$", size=30)
ax_z1.set_ylabel("Redshift$", size=30)
ax_z1.set_title("K-Means clustering", size=40)
ax_z1.set_xscale("log")
ax_z1.set_yscale("log")
ax_z1.legend(prop={"size": 30}, loc="lower left")

#Mean-shift labels

for i in np.unique(labels_ms):
    ax_z2.scatter(T90[labels_ms == i], redshift[labels_ms == i], color="C" + str(i), alpha=0.7) 

ax_z2.tick_params(labelsize=30)
ax_z2.set_xlabel("$T_{90}\ [s]$", size=30)
ax_z2.set_title("Mean-shift clustering", size=40)
ax_z2.set_xscale("log")
ax_z2.set_yscale("log")

#%%



