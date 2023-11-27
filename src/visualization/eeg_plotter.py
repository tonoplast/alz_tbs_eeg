# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:38:57 2023

@author: WookS
"""
# import mne

def plot_epochs(epochs):
    '''
    just cominbining all the plotting into one function
    '''
    epochs.plot()
    epochs.compute_psd(fmin=2, fmax=80).plot()
    epochs.compute_psd(fmin=2, fmax=80).\
        plot_topomap(bands = {'Delta (0-4 Hz)': (0, 4), 'Theta (4-8 Hz)': (4, 8),
         'Alpha (8-12 Hz)': (8, 12), 'Beta (12-30 Hz)': (12, 30),
         'Gamma (30-60 Hz)': (30, 60)})
    
    
