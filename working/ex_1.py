# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:22:02 2024

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt

N_doors = 3
win_count_s = 0
win_count_c = 0
win_count_n = 0
tot_choices = 0

#Generate the prizes

prize = ["Car"]

for i in range(0, N_doors - 1):
    prize.append("Goat")

#Run the code many times

for i in range(0, 10000):
    door = np.random.choice(prize, N_doors, replace=False)
    
    #Generate a choice for the switcher and conservative
    
    choice = door[np.random.randint(0, N_doors)]
    
    #Generate a choice for the newcomer
    
    new_door = np.random.choice(["Car", "Goat"], 2, replace=False)
    new_choice = new_door[np.random.randint(0, 2)]
    
    #The switcher loses if the first choice is a car, while the conservative wins
    
    if choice == "Car":
        win_count_c += 1
    
    if choice == "Goat":
        win_count_s += 1
        
    if new_choice == "Car":
        win_count_n += 1
        
    tot_choices += 1
    
    
#Probability computation
    
p_s = win_count_s / tot_choices
p_c = win_count_c / tot_choices
p_n = win_count_n / tot_choices
probs = [p_s, p_c, p_n]
    
#Draw an histogram with the probabilities
    
print("Switcher = " + str(p_s)) 
print("Conservative = " + str(p_c))   
print("Newcomer = " + str(p_n))      
    
plt.figure(figsize=(15, 15))
bar = plt.bar(["win_count_s", "win_count_c", "win_count_n"], height=[win_count_s, win_count_c, win_count_n], width=0.3, color=["green", "red", "blue"], label=["Switcher", "Conservative", "Newcomer"])
plt.xticks(size=30)
plt.yticks(size=30)
plt.legend(prop={"size": 30})

bar_i = 0

for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, f"{probs[bar_i]: .2%}", ha='center', va='bottom', size=30)
    bar_i += 1