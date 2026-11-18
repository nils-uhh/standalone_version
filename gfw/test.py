import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path

# Testdatei, um Sachen auszuprobieren - ansonsten keinerlei Relevanz 




def get_max_min(array):
    return max(array) , min(array)

def get_average(array):   
    return sum(array)/6 

                       
snr_levels = [0,5,10,15]  # hier die verschiedene SNR-Level eintragen
farben = ["blue","red","green","black"]
deltas = [0.2 , 0.4 , 0.6 , 0.8]
n = 0


array = [ [1,2,3,4,5,6],
         [0.1,0.2,0.3,0.4,0.5,0.6],
        [11,12,13,14,15,16],
        [1,1,1,1,1,1],
]

# Plotten
for i in range(len(deltas)): # for i in verschiedene delta
    punkte = []
    for j in range(len(snr_levels)): # verschiedene SNR-Level
        werte = array[j]
        max_min = get_max_min(werte)
        average = get_average(werte)

        x = snr_levels[j] 
        y =  average
        print(x)
        print(y)
        if j == 0:
            lbl = f"delta = {deltas[i]}"
        else:
            lbl = None

        punkte.append(y)
        plt.scatter(x,max_min[0],color=farben[n])
        plt.scatter(x,max_min[1],color=farben[n])
        plt.scatter(x,y,color=farben[n],label=lbl)
    plt.plot(snr_levels,punkte,color=farben[n])
    n = n + 1 


plt.xlim(0,30)
plt.ylim(0,20)
plt.ylabel("mean_loc_error in mm")
plt.xlabel("SNR in dB")
plt.title("Vergleich der Deltas")
plt.legend()
plt.show()


