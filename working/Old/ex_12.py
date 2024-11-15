# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:53:50 2024

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split

#Hand-identifed digits:

# 8 9 8 1 2 6 9
# 1 9 4 0 6 1 7
# 6 9 5 4 4 8 4
# 0 5 6 1 7 9 3
# 2 5 0 8 3 9 6
# 1 0 2 0 5 4 4
# 9 6 2 6 1 0 0

digits = datasets.load_digits()
print(digits.images.shape)
print(digits.keys())

fig, axes = plt.subplots(7,7, figsize=(10, 10))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

np.random.seed(42)
mychoices = np.random.choice(digits.images.shape[0],100)

for i, ax in enumerate(axes.flat):
    ax.imshow((digits.images[mychoices[i]]), 
              cmap='binary')
    ax.text(0.05, 0.05, str(digits.target[mychoices[i]]),transform=ax.transAxes, color='green', fontsize=14)
    ax.set_xticks([])
    

dig = digits["data"]

#%%------------------------------------------Isomap-----------------------------------------

iso = Isomap(n_neighbors=5, n_components=2)
iso.fit(dig)
red = iso.transform(dig)

plt.figure(figsize=(20, 20))
plt.scatter(red[:,0], red[:,1])

#%%