import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path

"""
Erstellt zu den simulierten Daten einen Plot, sodass schnell und übersichtlich verglichen werden kann.
"""

# Erstelle einen Plot, wobei jedes Delta einen eigenen Graph hat.
# jeder graph besteht aus einem SNR und entsprechendem mean_loc_error
# dadurch dann übersichtlich welches delta am besten ist
# delta = [0.2 , 0.4 , 0.6 , 0.8]??
# code hat mit delta = [0 , 1] irgendwie nicht funktioniert letztes mal  


def get_max_min(array):
    return max(array) , min(array)

def get_average(array):   
    return sum(array)/len(array) 

                       
snr_levels = [5,10,15,20]  # hier die verschiedene SNR-Level eintragen
farben = ["blue","red","green","black","yellow","purple"]     # jedes Delta hat eigene Farbe zur Unterscheidung
deltas = [0 , 0.2 , 0.4 , 0.6 , 0.8 , 1]
n = 0

# Plot vorbereiten und erstellen
for i in range(len(deltas)): # for i in verschiedene delta 
    punkte = []
    for j in range(len(snr_levels)): # verschiedene SNR-Level
        # Dateipfad finden:
        hauptordner = Path(__file__).resolve().parent
        pfad = (hauptordner / "Ergebnisse_sim_rep" / f"d = {deltas[i]}" / f"run_delta={deltas[i]}_SNR={snr_levels[j]}.npz") 
        # entsprechende Datei laden:
        with np.load(pfad) as archiv:
            werte = archiv["daten"]
        # Werte zum plotten bestimmen
        average = get_average(werte)
        max_min = get_max_min(werte)


        x = snr_levels[j] 
        y =  average

        if j == 0:  #   Jedes Delta eine eigene Farbe
            lbl = f"delta = {deltas[i]}"
        else:
            lbl = None
        punkte.append(y)
        # Erstellung des Plots

        plt.scatter(x,max_min[0],color=farben[n])
        plt.scatter(x,max_min[1],color=farben[n])
        plt.scatter(x,y,color=farben[n],label=lbl)

    plt.plot(snr_levels,punkte, color=farben[n])
    n = n + 1 


# Plotten

plt.xlim(0,30)  
plt.xlabel("SNR in dB")   

plt.ylim(0,100)
plt.ylabel("mean_loc_error in mm")

plt.title("Vergleich der Deltas")
plt.legend()
plt.show()
