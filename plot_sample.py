#import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_sample(N = 1,a = '1',b = 'a'):

    original_raman_data = pd.read_csv('data/'+a+b+'Raman.csv').values[:,1:]
    noised_raman_data = pd.read_csv('data/'+a+b+'CARS.csv').values[:,1:]
    peak_positions = pd.read_csv('data/'+a+b+'SW.csv').values[:,1:]
    denoised_raman_data = pd.read_csv('data_val/'+a+b+'Raman_spectrums_results.csv').values[:,1:]



    q=N
    plt.figure()
    plt.plot(peak_positions[q,:]/np.max(peak_positions[q,:]),c='yellow')
    plt.plot(noised_raman_data[q,:],c='r',linewidth=2.0)
    plt.plot(denoised_raman_data[q,:],c='deepskyblue',linewidth=3.0)
    plt.plot(original_raman_data[q,:],c='k')
    plt.grid()
    plt.show()
