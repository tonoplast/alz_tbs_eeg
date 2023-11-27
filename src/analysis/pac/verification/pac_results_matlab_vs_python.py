# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 22:09:09 2023

@author: sungw
"""

from src.analysis.pac.verification.pac_results_comparator import get_differences_between_the_two_data, test_for_difference, stripplot_comparison, get_distance
from pathlib import Path
import scipy.io
import pandas as pd
import pingouin


# =============================================================================
# python
# =============================================================================

# eyes open: wavelet vs hilbert
drive = 'D:'
data_folder_1 = Path(f'{drive}/tram_protocol_eeg/resting_relax/RELAXProcessed/Cleaned_Data/eyes_open/output/tensorpac')
data_folder_2 = Path(f'{drive}/tram_protocol_eeg/resting_relax/RELAXProcessed/Cleaned_Data/eyes_open/output/tensorpac')
wildcard1 = '*_eo_wavelet*.json'
wildcard2 = '*_eo_hilbert*.json'
x1 = 'filtertype'
x2 = 'filtertype'
y1 = 'peak_theta'
y2 = 'peak_gamma'
filename_split_colnames = ['id','session','timepoint','eyestatus','filtertype']
pair_cols = ['id','session','timepoint','eyestatus']

ttests, means, stds, df_wavelet_eo, df_hilbert_eo = get_differences_between_the_two_data(data_folder_1, data_folder_2, wildcard1, wildcard2, x1, y1, x2, y2, filename_split_colnames, pair_cols = pair_cols)



# eyes closed: wavelet vs hilbert
drive = 'D:'
data_folder_1 = Path(f'{drive}/tram_protocol_eeg/resting_relax/RELAXProcessed/Cleaned_Data/eyes_closed/output/tensorpac')
data_folder_2 = Path(f'{drive}/tram_protocol_eeg/resting_relax/RELAXProcessed/Cleaned_Data/eyes_closed/output/tensorpac')
x1 = 'filtertype'
x2 = 'filtertype'
y1 = 'peak_theta'
y2 = 'peak_gamma'
wildcard1 = '*_ec_wavelet*.json'
wildcard2 = '*_ec_hilbert*.json'
filename_split_colnames = ['id','session','timepoint','eyestatus','filtertype']
pair_cols = ['id','session','timepoint','eyestatus']

ttests, means, stds, df_wavelet_ec, df_hilbert_ec = get_differences_between_the_two_data(data_folder_1, data_folder_2, wildcard1, wildcard2, x1, y1, x2, y2, filename_split_colnames, pair_cols = pair_cols)


#  wavelet: eyes open vs eyes closed
drive = 'D:'
data_folder_1 = Path(f'{drive}/tram_protocol_eeg/resting_relax/RELAXProcessed/Cleaned_Data/eyes_open/output/tensorpac')
data_folder_2 = Path(f'{drive}/tram_protocol_eeg/resting_relax/RELAXProcessed/Cleaned_Data/eyes_closed/output/tensorpac')
x1 = 'eyestatus'
x2 = 'eyestatus'
y1 = 'peak_theta'
y2 = 'peak_gamma'
wildcard1 = '*_wavelet*.json'
wildcard2 = '*_wavelet*.json'
filename_split_colnames = ['id','session','timepoint','eyestatus','filtertype']
pair_cols = ['id','session','timepoint']

ttests, means, stds, df1, df2 = get_differences_between_the_two_data(data_folder_1, data_folder_2, wildcard1, wildcard2, x1, y1, x2, y2, filename_split_colnames, pair_cols = pair_cols)



#  hilbert: eyes open vs eyes closed
drive = 'D:'
data_folder_1 = Path(f'{drive}/tram_protocol_eeg/resting_relax/RELAXProcessed/Cleaned_Data/eyes_open/output/tensorpac')
data_folder_2 = Path(f'{drive}/tram_protocol_eeg/resting_relax/RELAXProcessed/Cleaned_Data/eyes_closed/output/tensorpac')
x1 = 'eyestatus'
x2 = 'eyestatus'
y1 = 'peak_theta'
y2 = 'peak_gamma'
wildcard1 = '*_hilbert*.json'
wildcard2 = '*_hilbert*.json'
filename_split_colnames = ['id','session','timepoint','eyestatus','filtertype']
pair_cols = ['id','session','timepoint']

ttests, means, stds, df1, df2 = get_differences_between_the_two_data(data_folder_1, data_folder_2, wildcard1, wildcard2, x1, y1, x2, y2, filename_split_colnames, pair_cols = pair_cols)






# =============================================================================
# matlab
# =============================================================================
matlab_eo = scipy.io.loadmat(Path(f'{drive}/tram_protocol_eeg/resting_relax/RELAXProcessed/Cleaned_Data/eyes_open/output/glmf/matlab_eo.mat')).get('peak_freqs')
df_matlab_eo = pd.DataFrame(matlab_eo, columns=['peak_theta', 'peak_gamma'])
df_matlab_eo[['id', 'session', 'timepoint']] = df1[['id', 'session', 'timepoint']].copy()
df_matlab_eo['filtertype'] = 'glmf'
df_matlab_eo['eyestatus'] = 'eo'

matlab_ec = scipy.io.loadmat(Path(f'{drive}/tram_protocol_eeg/resting_relax/RELAXProcessed/Cleaned_Data/eyes_closed/output/glmf/matlab_ec.mat')).get('peak_freqs')
df_matlab_ec = pd.DataFrame(matlab_ec, columns=['peak_theta', 'peak_gamma'])
df_matlab_ec[['id', 'session', 'timepoint']] = df1[['id', 'session', 'timepoint']].copy()
df_matlab_ec['filtertype'] = 'glmf'
df_matlab_ec['eyestatus'] = 'ec'

ttests_matlab_eo_vs_ec, means_matlab_eo_vs_ec, stds_matlab_eo_vs_ec = test_for_difference(df_matlab_eo, df_matlab_ec, has_sig=False)



# =============================================================================
# matlab vs python (theta and gamma separately)
# =============================================================================

# glmf (matlab) eo vs ec
df = pd.concat([df_matlab_eo, df_matlab_ec])
stripplot_comparison(x='eyestatus', y='peak_theta', df=df, title='glmf (matlab)', pair_cols = ['id','session','timepoint'])
stripplot_comparison(x='eyestatus', y='peak_gamma', df=df, title='glmf (matlab)', pair_cols = ['id','session','timepoint'])


# glmf (matlab) vs wavelet (python) - eo
df = pd.concat([df_matlab_eo, df_wavelet_eo])
stripplot_comparison(x='filtertype', y='peak_theta', df=df, title='eo', pair_cols = ['id','session','timepoint'])
stripplot_comparison(x='filtertype', y='peak_gamma', df=df, title='eo', pair_cols = ['id','session','timepoint'])
ttests_eo_matlab_vs_python_wavelet, means_eo_matlab_vs_python_wavelet, stds_eo_matlab_vs_python_wavelet = test_for_difference(df_matlab_eo, df_wavelet_eo, has_sig=False)


# glmf (matlab) vs wavelet (python) - ec
df = pd.concat([df_matlab_ec, df_wavelet_ec])
stripplot_comparison(x='filtertype', y='peak_theta', df=df, title='ec')
stripplot_comparison(x='filtertype', y='peak_gamma', df=df, title='ec')
ttests_ec_matlab_vs_python_wavelet, means_ec_matlab_vs_python_wavelet, stds_ec_matlab_vs_python_wavelet = test_for_difference(df_matlab_ec, df_wavelet_ec, has_sig=False)


# glmf (matlab) vs hilbert (python) - eo
df = pd.concat([df_matlab_eo, df_hilbert_eo])
stripplot_comparison(x='filtertype', y='peak_theta', df=df, title='eo')
stripplot_comparison(x='filtertype', y='peak_gamma', df=df, title='eo')
ttests_eo_matlab_vs_python_hilbert, means_eo_matlab_vs_python_hilbert, stds_eo_matlab_vs_python_hilbert = test_for_difference(df_matlab_eo, df_hilbert_eo, has_sig=False)


# glmf (matlab) vs hilbert (python) - ec
df = pd.concat([df_matlab_ec, df_hilbert_ec])
stripplot_comparison(x='filtertype', y='peak_theta', df=df, title='ec')
stripplot_comparison(x='filtertype', y='peak_gamma', df=df, title='ec')
ttests_ec_matlab_vs_python_hilbert, means_ec_matlab_vs_python_hilbert, stds_ec_matlab_vs_python_hilbert = test_for_difference(df_matlab_ec, df_hilbert_ec, has_sig=False)



# =============================================================================
# matlab vs python (distance)
# =============================================================================

# wavelet vs hilbert eo distance from glmf
dist_matlab_wavelet_eo = get_distance(df_matlab_eo, df_wavelet_eo, 'peak_theta', 'peak_gamma')
dist_matlab_hilbert_eo = get_distance(df_matlab_eo, df_hilbert_eo, 'peak_theta', 'peak_gamma')
print('\n[Mean Distance (the lower the better)]')
print(f'matlab_to_wavelet-eo: {round(dist_matlab_wavelet_eo.mean(), 2)}\nmatlab_to_hilbert-eo: {round(dist_matlab_hilbert_eo.mean(), 2)}')
ttest_dist_eo = pingouin.ttest(dist_matlab_wavelet_eo, dist_matlab_hilbert_eo, paired=True)


df_glmf_and_wavelet_distance = pd.DataFrame(data=dist_matlab_wavelet_eo, columns=['distance'])
df_glmf_and_wavelet_distance['filtertype'] = 'glmf_vs_wavelet'
df_glmf_and_wavelet_distance[['id','session','timepoint','eyestatus']] = df_wavelet_eo[['id','session','timepoint','eyestatus']]

df_glmf_and_hilbert_distance = pd.DataFrame(data=dist_matlab_hilbert_eo, columns=['distance'])
df_glmf_and_hilbert_distance['filtertype'] = 'glmf_vs_hilbert'
df_glmf_and_hilbert_distance[['id','session','timepoint','eyestatus']] = df_hilbert_eo[['id','session','timepoint','eyestatus']]



df = pd.concat([df_glmf_and_wavelet_distance, df_glmf_and_hilbert_distance])
stripplot_comparison(x='filtertype', y='distance', df=df, title='eo', pair_cols = ['id','session','timepoint'])


# wavelet vs hilbert ec distance from glmf
dist_matlab_wavelet_ec = get_distance(df_matlab_ec, df_wavelet_ec, 'peak_theta', 'peak_gamma')
dist_matlab_hilbert_ec = get_distance(df_matlab_ec, df_hilbert_ec, 'peak_theta', 'peak_gamma')
print('\n[Mean Distance (the lower the better)]')
print(f'matlab_to_wavelet-ec: {round(dist_matlab_wavelet_ec.mean(), 2)}\nmatlab_to_hilbert-ec: {round(dist_matlab_hilbert_ec.mean(), 2)}')
ttest_dist_ec = pingouin.ttest(dist_matlab_wavelet_ec, dist_matlab_hilbert_ec, paired=True)

df_glmf_and_wavelet_distance = pd.DataFrame(data=dist_matlab_wavelet_ec, columns=['distance'])
df_glmf_and_wavelet_distance['filtertype'] = 'glmf_vs_wavelet'
df_glmf_and_wavelet_distance[['id','session','timepoint','eyestatus']] = df_wavelet_ec[['id','session','timepoint','eyestatus']]

df_glmf_and_hilbert_distance = pd.DataFrame(data=dist_matlab_hilbert_ec, columns=['distance'])
df_glmf_and_hilbert_distance['filtertype'] = 'glmf_vs_hilbert'
df_glmf_and_hilbert_distance[['id','session','timepoint','eyestatus']] = df_hilbert_ec[['id','session','timepoint','eyestatus']]

df = pd.concat([df_glmf_and_wavelet_distance, df_glmf_and_hilbert_distance])
stripplot_comparison(x='filtertype', y='distance', df=df, title='ec', pair_cols = ['id','session','timepoint'])
