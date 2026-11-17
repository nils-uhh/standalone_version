import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path


# Erstelle einen Plot, wobei jedes Delta einen eigenen Graph hat.
# jeder graph besteht aus einem SNR und entsprechendem mean_loc_error
# dadurch dann übersichtlich welches delta am besten ist
# delta = [0.2 , 0.4 , 0.6 , 0.8]??
# code hat mit delta = [0 , 1] irgendwie nicht funktioniert letztes mal  


def get_max_min(array):
    return max(array) , min(array)

def get_average(array):   
    return sum(array)/100 


snr_levels = [1,2,3,4]
farben = ["blue","red","green","black"]
n = 0

# Plotten
for i in [0.2 , 0.4 , 0.6 , 0.8]: # for i in verschiedene delta 
    for j in [snr_levels]: # verschiedene SNR-Level
        # Dateipfad finden:
        hauptordner = Path(__file__).resolve().parent
        pfad = (hauptordner / "Ergebnisse_sim_rep" / f"d = {i}" / "run_delta={i}_SNR={j}.npz") 
        # entsprechende Datei laden:
        with np.load(pfad) as archiv:
            werte = archiv["daten"]
        # Werte zum plotten bestimmen
        average = get_average(werte)
        max_min = get_max_min(werte)


        x = j 
        y =  average
        plt.scatter(x,y,color=farben[n],label=f"delta = {i}")

    n = n + 1 


plt.xlim(0,40)
plt.ylim(0,100)
plt.ylabel("mean_loc_error in mm")
plt.xlabel("SNR in dB")
plt.title("Vergleich der Deltas")
plt.legend()
plt.show()
