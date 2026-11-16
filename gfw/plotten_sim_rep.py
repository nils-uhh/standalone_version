import matplotlib 
import matplotlib.pyplot as plt
import numpy as np

# Erstelle einen Plot, wobei jedes Delta einen eigenen Graph hat.
# jeder graph besteht aus einem SNR und entsprechendem mean_loc_error
# dadurch dann übersichtlich welches delta am besten ist
# delta = [0.2 , 0.4 , 0.6 , 0.8]??
# code hat mit delta = [0 , 1] irgendwie nicht funktioniert letztes mal  


array1 = [1,2,3,4,5,6,7,] # muss dann später dynamisch aus Ergebnisse_sim_rep geholt werden
array2 = [11,12,13,14,15,16]


def get_average(npz):   
    return sum[npz]/100 

def get_max_min(npz):
    return max(npz) , min(npz)





# Plotten

for i in 00: # for i in verschiedene delta 
    snr_level = None # hier die verschiedenen getesteten snr_level eintragen 
    x = [snr_level] 
    y = get_average("run_delta:{delta}_SNR:{snr}.npz")  # so ungefähr vom vibe her 
    plt.plot(x,y,color="")

plt.ylabel("mean_loc_error in mm")
plt.xlabel("SNR in db")
plt.title("Vergleich der Deltas")
plt.legend()




plt.show()
