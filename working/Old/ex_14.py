# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:57:14 2024

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import urllib.request
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

#----------Read data----------

urllib.request.urlretrieve("https://raw.githubusercontent.com/nshaud/ml_for_astro/main/stars.csv", "stars.csv")

df_stars = pd.read_csv("stars.csv")

#----------Transform star types and star colors to numbers----------

le = LabelEncoder()
# Assign unique integers from 0 to 6 to each star type
df_stars['Star type'] = le.fit_transform(df_stars['Star type'])
labels = le.inverse_transform(df_stars['Star type'])
class_names = le.classes_
print(class_names)

le_c = LabelEncoder()
df_stars['Star color'] = le_c.fit_transform(df_stars['Star color'])
color_names = le_c.classes_
print(color_names)

#----------Clean data----------

cond_missing = ~np.isnan(np.log(df_stars["Temperature (K)"])) & ~np.isnan(np.log(df_stars["Luminosity(L/Lo)"])) & ~np.isnan(np.log(df_stars["Radius(R/Ro)"]))

T = np.array(df_stars["Temperature (K)"][cond_missing])
L = np.array(df_stars["Luminosity(L/Lo)"][cond_missing])
R = np.array(df_stars["Radius(R/Ro)"][cond_missing])
Mv = np.array(df_stars["Absolute magnitude(Mv)"][cond_missing])

#----------Plot----------

fig = plt.figure(figsize=(7, 7))
sns.scatterplot(data=df_stars, x='Temperature (K)', y='Luminosity(L/Lo)', hue=labels)

plt.xscale('log')
plt.yscale('log')
plt.xticks([5000, 10000, 50000])
plt.xlim(5e4, 1.5e3)
plt.show()

#%%-----------------------------Principal Component Analysis-------------------------------------------

#----------Prepare the data----------

X = np.stack((T, L, R, Mv), axis=1)

for i in range(0, len(X[0])):
    X[:, i] = (X[:, i]  / np.std(X[:, i])**2) - np.mean(X[:, i]  / np.std(X[:, i])**2)

#----------Cross validation for the PCA----------

cv = GridSearchCV(PCA(), {"n_components": [1, 2, 3, 4]}, cv=len(X), verbose=3)
cv.fit(X)
best_comp = cv.best_params_["n_components"]
print(best_comp)

#----------PCA----------

pca = PCA(n_components=best_comp)
pca.fit(X)
e_values = pca.transform(X)
e_vectors = pca.components_
mean = pca.mean_
var_r = pca.explained_variance_ratio_

#----------Scree plots----------

fig_scree = plt.figure(figsize=(20, 10))
ax_scree_1 = fig_scree.add_subplot(121)
ax_scree_2 = fig_scree.add_subplot(122)

ax_scree_1.plot(np.arange(0, best_comp), var_r, marker="o", markersize=10)
ax_scree_2.plot(np.arange(0, best_comp), np.cumsum(var_r), marker="o", markersize=10)

#----------Dataset projection on principal axes----------

proj = []

for i in range(0, best_comp):
    proj.append(np.dot(e_values, e_vectors[:, i]))
    
proj = np.array(proj)
    
plt.figure(figsize=(20, 20))
plt.scatter(proj[0], proj[1])
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlabel("Component 0", size=30)
plt.ylabel("Component 1", size=30)
plt.gca().invert_xaxis()
# plt.xscale("log")
# plt.yscale("log")

#%%



