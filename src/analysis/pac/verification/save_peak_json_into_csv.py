# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:04:02 2023

@author: WookS
"""

from pathlib import Path
import scipy.io
import pandas as pd
import numpy as np
import pingouin
import seaborn as sns
import matplotlib.pyplot as plt
from src.analysis.pac.verification.pac_results_comparator import df_wrapper

DRIVE = 'O:'
MAIN_FOLDER = r'\\CogTx\\DataTransfer\\ALZ TBS\\e. EEG data\\TGC\\processed_EEG\\'

# COMPARE_FOLDERS = ['025Hz_MWF', '025Hz_NoMWF', '1Hz_MWF', '1Hz_NoMWF']

COMPARE_FOLDERS = ['025Hz_NoMWF_ALL']
CENTRAL_TENDENCIES = ['mean', 'median']

dfs = []

for COMPARE_FOLDER in COMPARE_FOLDERS:
    for CENTRAL_TENDENCY in CENTRAL_TENDENCIES:

        DATA_PATH = Path(f'{DRIVE}/{MAIN_FOLDER}/{COMPARE_FOLDER}/RELAXProcessed/Cleaned_Data/Epoched/results_pac/{CENTRAL_TENDENCY}')
        OUTPUT_PATH = DATA_PATH.joinpath(f'output_{COMPARE_FOLDER}.csv')   
        df_csv = df_wrapper(DATA_PATH, '*.json', filename_split_colnames=['id','eyestatus','timepoint','agg_type'], x_user_defined='')
        df_csv.to_csv(OUTPUT_PATH)
        dfs.append(df_csv)
    

# replace 0 with nan
[df.replace(0, np.nan, inplace=True) for df in dfs]


check = [df[['peak_theta', 'peak_gamma','sig_pixel_theta','sig_pixel_gamma','agg_type']].groupby('agg_type').mean(numeric_only=True) for df in dfs]
