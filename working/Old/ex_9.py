# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:04:35 2024

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import corner

#%%---------------------------------Generate the chain-------------------------------------

N = 100000

def WeatherChain(day_initial, N):
    day_current = [day_initial]
    
    for i in range(0, N - 1):
        if day_current[i] == 0:
            if np.random.uniform(0, 1) >= 0.5:
                day_current.append(0)
            
            else:
                day_current.append(1)

        elif day_current[i] == 1:
            if np.random.uniform(0, 1) >= 0.1:
                day_current.append(1)
                
            else:
                day_current.append(0)
            
    return np.array(day_current)


days = WeatherChain(0, N)

tot_sunny = []

for i in range(1, N):
    tot_sunny.append(len(days[:i-1][days[:i-1] == 0]) / i)
    
tot_sunny = np.array(tot_sunny)

#%%

#%%----------------------------Plot the number of sunny days-------------------------------------
    
plt.figure(figsize=(20, 20))
plt.plot(np.arange(1, N), tot_sunny)
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlabel("Days", size=30)
plt.ylabel("Total sunny days / total days", size=30)

plt.figure(figsize=(20, 20))
plt.hist(tot_sunny, "fd")
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlabel("$p$(clear)", size=30)
plt.ylabel("Distribution of $p$(clear)", size=30)
plt.xlim(0.14, 0.20)

#%%

#%%------------------------------------Summary statistics----------------------------------

best_days = tot_sunny.mean()
best_days_std = tot_sunny.std() / np.sqrt(len(tot_sunny))

print("Mean = " + f"{best_days: .3f}")
print("Std = " + f"{best_days_std: .3e}")
    
#%%