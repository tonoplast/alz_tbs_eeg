# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import scipy
from meegkit import dss
from meegkit.detrend import regress, detrend
from meegkit.utils import create_line_data, unfold
from scipy import signal
from autoreject import get_rejection_threshold
from autoreject import AutoReject
from mne_icalabel import label_components

# sample_data_folder = mne.datasets.sample.data_path()
# sample_data_raw_file = (sample_data_folder / 'MEG' / 'sample' /
#                         'sample_audvis_filt-0-40_raw.fif')
# raw = mne.io.read_raw_fif(sample_data_raw_file)


p = Path('.').resolve()

test_data_folder = p.joinpath('data','H309')
test_data_raw_file = test_data_folder.joinpath("S50_T0_resting.cnt")
raw = mne.io.read_raw_cnt(test_data_raw_file, preload=True)
# raw = mne.io.read_raw_cnt(test_data_raw_file)

print(raw)
print(raw.info)

# dropping (unused) channels
dropchans = ['FP1','FPZ','FP2','FT7','FT8','TP7', 'TP8','PO5','PO6','PO7','PO8',
             'CB1','CB2','E1','E3','HEOG','CPZ','M1','M2']

raw.drop_channels(dropchans, on_missing='raise')


sfreq = raw.info['sfreq']


# =============================================================================
# # filter (highpass/lowpass/spike)
# =============================================================================
high_pass = 1 # since not erp, it should be okay, and good for ICA
low_pass  = 100
# raw.plot_psd(fmax=80)
raw_filt = raw.filter(high_pass, low_pass, fir_design='firwin')      
# raw.plot_psd(fmax=80)



# =============================================================================
# ## notch filter
# =============================================================================
raw_50hz_removed = raw_filt.copy().notch_filter(np.arange(50, 251, 50), 
                                    filter_length='auto', 
                                    phase='zero',
                                    method='spectrum_fit')

raw_filt.plot_psd(fmax=100)
raw_50hz_removed.plot_psd(fmax=100)

# =============================================================================
# ## detrend?
# =============================================================================
raw_detrend = raw_50hz_removed.copy()

for i in range(raw.info['nchan']):
    channel_data = raw_detrend._data[i, :]
    detrended_data, _, _ = detrend(channel_data, order=1)
    raw_detrend._data[i, :] = detrended_data


# # baseline correction for entire period?
# # Define the baseline period
# tmin, tmax = raw_detrend.times[0], raw_detrend.times[-1]  # use the entire data as the baseline

# # Apply the baseline correction
# raw_detrend.apply_baseline((tmin, tmax), method='demean')


# =============================================================================
# # baseline correction with entire data (demean)
# =============================================================================
# raw_detrend.plot(duration=5, n_channels=len(raw_detrend.info.ch_names))
data = raw_detrend.get_data()
# Demean the data
data = data - np.mean(data, axis=1, keepdims=True)
# Update the raw object with the demeaned data
raw_detrend._data = data
# raw_detrend.plot(duration=5, n_channels=len(raw_detrend.info.ch_names))


# =============================================================================
# # average reference
# =============================================================================
raw_avg_ref = raw_detrend.copy().set_eeg_reference(ref_channels='average')
raw_avg_ref.plot_psd(fmax=100)
raw_avg_ref.plot(duration=5, n_channels=len(raw.info.ch_names))


# =============================================================================
# # ICA on raw data (infomax)
# =============================================================================
ica = mne.preprocessing.ICA(n_components=0.99, 
                            random_state=99, 
                            max_iter='auto',
                            method='infomax',
                            fit_params=dict(extended=True))
ica.fit(raw_avg_ref)

ic_labels = label_components(raw_avg_ref, ica, method='iclabel')
labels = ic_labels["labels"]
exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
print(f"Excluding these ICA components via auto-labelling: {exclude_idx}")



# Exclude blink artifact components (use FP1/FP2 as EOG proxy)
epochs_eog    = mne.preprocessing.create_eog_epochs(raw = raw, ch_name = ['AF3','AF4']) 
_, eog_scores = ica.find_bads_eog(epochs_eog, ch_name =   ['AF3','AF4'], measure = 'zscore')
exclude_components = [np.argwhere(abs(i) > 0.5).ravel().tolist() for i in eog_scores]
exclude_components = list(set(sum(exclude_components, [])))
print(f"Excluding blink-related ICA components: {exclude_components}")

exclude_idx_all = list(set(exclude_idx + exclude_components))
ica.exclude   = exclude_idx_all

reconst_raw = raw_avg_ref.copy()
ica.apply(raw_avg_ref, exclude=ica.exclude)


raw_avg_ref.plot(show_scrollbars=False)
reconst_raw.plot(show_scrollbars=False)

# =============================================================================
# epoch continuous data
# =============================================================================
tmin = 0
tmax = 12
overlap = 9

events = mne.make_fixed_length_events(reconst_raw, id=1, duration=tmax-tmin, overlap=overlap)
epochs = mne.Epochs(reconst_raw, 
                    events, 
                    tmin=tmin, 
                    tmax=tmax, 
                    baseline=(None,None),
                    # baseline=None,
                    preload=True)


# =============================================================================
# Autoreject 1
# =============================================================================

ar = AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
                           n_jobs=1, verbose=True)
ar.fit(epochs[:20])  # fit on a few epochs to save time
epochs_ar, reject_log = ar.transform(epochs, return_log=True)

# visualise
epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))
reject_log.plot('horizontal')


# plot with and without ICA exclusion
ica.plot_overlay(epochs.average(), exclude=ica.exclude)
# ica.apply(epochs, exclude=ica.exclude)


# ar = AutoReject()
# raw_ica = ar.fit_transform(epochs)  




# =============================================================================
# check
# =============================================================================
evoked_bad = epochs[reject_log.bad_epochs].average()
plt.figure()
plt.plot(evoked_bad.times, evoked_bad.data.T * 1e6, 'r', zorder=-1)
epochs_ar.average().plot(axes=plt.gca())


# chs = ['AF3','AF4','F1','FZ','F2']#,'C1','CZ','C2','P1','PZ','P2','O1','OZ','O2']
# chan_idxs = [epochs.ch_names.index(ch) for ch in chs]
# epochs.plot(order=chan_idxs)
