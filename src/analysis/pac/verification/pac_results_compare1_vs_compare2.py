# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 22:09:09 2023

@author: sungw
"""

from src.analysis.pac.verification.pac_results_comparator import get_differences_between_the_two_data
from src.analysis.pac.verification.pac_results_comparator import test_for_difference
from src.analysis.pac.verification.pac_results_comparator import stripplot_comparison
from src.analysis.pac.verification.pac_results_comparator import get_distance
from src.analysis.pac.verification.pac_results_comparator import get_epoch_number_df
from src.analysis.pac.verification.pac_results_comparator import run_multiple_regression_for_epoch_number

from pathlib import Path
import scipy.io
import pandas as pd
import pingouin
import seaborn as sns
import matplotlib.pyplot as plt

# import json


# =============================================================================
# python
# =============================================================================

### MAKE SURE TO RUN 'get_epoch_numbers.py' before comparing to get the epoch numbers ###      
MAIN_FOLDER = r'\EEG_data_collected_today\processed_EEG'
# MAIN_FOLDER = r'\\Cognitive Disorders\\Data Transfer\\ALZ TBS\\e. EEG data\\TGC\\processed_EEG\\'


which_compare_iteration = 40

which_comparison = f'comp{which_compare_iteration}'

if (which_compare_iteration >=1) & (which_compare_iteration >=8):
    which_matlab_comparison = '1Hz_HighPass_MWF_01lostep_1bw'
elif (which_compare_iteration >=9) & (which_compare_iteration >=17):
    which_matlab_comparison = '025Hz_HighPass_MWF_01lostep_1bw'
else:
    which_matlab_comparison = '025Hz_HighPass_MWF_01lostep_1bw'


comparison_record = {
    
    ## 1Hz theta
    'comp1': ['1Hz_HighPass_MWF_01lostep_1bw', '1Hz_HighPass_MWF_01lostep_2bw'],
    'comp2': ['1Hz_HighPass_MWF_01lostep_1bw', '1Hz_HighPass_MWF'],
    'comp3': ['1Hz_HighPass_MWF', '1Hz_HighPass_MWF_18s'],
    'comp4': ['1Hz_HighPass_MWF', '1Hz_HighPass_NoMWF'],
    'comp5': ['1Hz_HighPass_MWF', '1Hz_HighPass_MWF_Zapline'],
    'comp6': ['1Hz_HighPass_MWF', '1Hz_HighPass_NoMWF_Zapline'],
    'comp7': ['1Hz_HighPass_MWF', '1Hz_HighPass_MWF_MI'],
    'comp8': ['1Hz_HighPass_MWF', '1Hz_HighPass_MWF_F1FZF2P1PZP2'],
    
    ## 0.25 Hz theta
    'comp9': ['025Hz_HighPass_MWF_01lostep_1bw', '025Hz_HighPass_MWF_01lostep_2bw'],
    'comp10': ['025Hz_HighPass_MWF_01lostep_1bw', '025Hz_HighPass_MWF'],
    'comp11': ['025Hz_HighPass_MWF', '025Hz_HighPass_MWF_18s'],
    'comp12': ['025Hz_HighPass_MWF', '025Hz_HighPass_NoMWF'],
    'comp13': ['025Hz_HighPass_MWF', '025Hz_HighPass_MWF_Zapline'],
    'comp14': ['025Hz_HighPass_MWF', '025Hz_HighPass_NoMWF_Zapline'],
    'comp15': ['025Hz_HighPass_MWF', '025Hz_HighPass_MWF_MI'],
    'comp16': ['025Hz_HighPass_MWF', '025Hz_HighPass_MWF_F1FZF2P1PZP2'],
    'comp17': ['025Hz_HighPass_MWF', '025Hz_HighPass_MWF_Laplacian'],


    # 1 Hz vs 0.25 Hz
    'comp18': ['1Hz_HighPass_MWF', '025Hz_HighPass_MWF'],
    'comp19': ['1Hz_HighPass_MWF_01lostep_1bw', '025Hz_HighPass_MWF_01lostep_1bw'],
    'comp20': ['1Hz_HighPass_MWF_01lostep_2bw', '025Hz_HighPass_MWF_01lostep_2bw'],
    'comp21': ['1Hz_HighPass_MWF_18s', '025Hz_HighPass_MWF_18s'],
    'comp22': ['1Hz_HighPass_NoMWF', '025Hz_HighPass_NoMWF'],
    'comp23': ['1Hz_HighPass_MWF_Zapline', '025Hz_HighPass_MWF_Zapline'],
    'comp24': ['1Hz_HighPass_NoMWF_Zapline', '025Hz_HighPass_NoMWF_Zapline'],
    'comp25': ['1Hz_HighPass_MWF_MI', '025Hz_HighPass_MWF_MI'],
    'comp26': ['1Hz_HighPass_MWF_F1FZF2P1PZP2', '025Hz_HighPass_MWF_F1FZF2P1PZP2'],
    
    # other for interest
    'comp27': ['1Hz_HighPass_MWF_Zapline', '1Hz_HighPass_NoMWF_Zapline'],
    'comp28': ['025Hz_HighPass_MWF_Zapline', '025Hz_HighPass_NoMWF_Zapline'],

    'comp29': ['1Hz_HighPass_MWF_Zapline', '025Hz_HighPass_MWF_Zapline'],
    'comp30': ['1Hz_HighPass_NoMWF_Zapline', '025Hz_HighPass_NoMWF_Zapline'],
    
    'comp31': ['025Hz_HighPass_NoMWF_Zapline', '025Hz_HighPass_NoMWF_Zapline_Laplacian_F1FZF2P1PZP2'],
    'comp32': ['025Hz_HighPass_NoMWF_Zapline_Laplacian_F1FZF2P1PZP2', '025Hz_HighPass_NoMWF_Zapline_Laplacian_FZPZ_1'],
    'comp33': ['025Hz_HighPass_NoMWF_Zapline_Laplacian_FZPZ_1', '025Hz_HighPass_NoMWF_Zapline_Laplacian_FZPZ_2'],
    'comp34': ['025Hz_HighPass_NoMWF_Zapline_Laplacian_F1FZF2P1PZP2', '025Hz_HighPass_NoMWF_Zapline_Laplacian_FZPZ_2'],

    'comp35': ['025Hz_HighPass_NoMWF_Zapline', '025Hz_HighPass_NoMWF_Zapline_Laplacian_FZPZ_2'],
    'comp36': ['025Hz_HighPass_NoMWF_Zapline_Laplacian_FZPZ_1', '025Hz_HighPass_NoMWF_Zapline_Laplacian_FZPZ_2'],
    'comp37': ['025Hz_HighPass_MWF_Zapline_Laplacian_FZPZ', '025Hz_HighPass_NoMWF_Zapline_Laplacian_FZPZ_1'],
    'comp38': ['025Hz_HighPass_MWF_Zapline_Laplacian_FZPZ', '025Hz_HighPass_NoMWF_Zapline_Laplacian_FZPZ_2'],
    
    'comp39': ['025Hz_HighPass_MWF_Zapline_Laplacian_FZPZ', '025Hz_HighPass_MWF_Zapline_Laplacian_FZPZ_02epoch'],
    'comp40': ['025Hz_HighPass_MWF_Laplacian_FZPZ_higher_threshold', '025Hz_HighPass_MWF_Laplacian_FZPZ_median'],

    }

compare1 = comparison_record.get(which_comparison)[0]
compare1_folder = compare1

compare2 = comparison_record.get(which_comparison)[1]
compare2_folder = compare2

DRIVE = 'E:'
COMPARE1_PATH = Path(f'{DRIVE}/{MAIN_FOLDER}/{compare1_folder}/RELAXProcessed/Cleaned_data/Epoched/results_pac')
COMPARE2_PATH = Path(f'{DRIVE}/{MAIN_FOLDER}/{compare2_folder}/RELAXProcessed/Cleaned_data/Epoched/results_pac')

COMPARE1_EPOCH_NUMBER_PATH = Path(f'{DRIVE}/{MAIN_FOLDER}/{compare1_folder}/RELAXProcessed/Cleaned_data/Epoched/eoec')
COMPARE2_EPOCH_NUMBER_PATH = Path(f'{DRIVE}/{MAIN_FOLDER}/{compare2_folder}/RELAXProcessed/Cleaned_data/Epoched/eoec')


# =============================================================================
# compare epochs / frequencies (pre-processing)
# =============================================================================

try:
    df_compare1_epoch_number, compare1_mean_std, eo1, ec1 = get_epoch_number_df(COMPARE1_EPOCH_NUMBER_PATH)
    df_compare2_epoch_number, compare2_mean_std, eo2, ec2 = get_epoch_number_df(COMPARE2_EPOCH_NUMBER_PATH)
    
    ttest_epoch_number = pingouin.ttest(eo1['epoch_number'], eo2['epoch_number'], paired=True)
    
    eo1['comparison'] = compare1
    eo2['comparison'] = compare2
    df_long = pd.concat([eo1, eo2]).drop(columns=['filename','eye_status'])
    df_long = pd.melt(df_long, id_vars=['epoch_number', 'comparison'], var_name='freq', value_name='power')
    
    
    fig, ax = plt.subplots(figsize=(9,5))
    sns.boxplot(data=df_long, x='freq', y='power', hue='comparison' )
    plt.yscale('log')
    sns.despine()
    plt.show()
    
    fig, ax = plt.subplots(figsize=(9,5))
    sns.boxplot(data=df_long, y="epoch_number", x='comparison')
    sns.despine()
    plt.show()
except:
    print('Data not available.')


### SER / ARR / other cleaned metrics

def extract_cleaned_metric_of_interest(matlab_metric_path, which_metric):
    '''
    Getting cleaned metrics from RELAX based on the metric we want to extract
    '''
    matlab_cleaned_metrics = scipy.io.loadmat(matlab_metric_path.joinpath('CleanedMetrics.mat')).get('CleanedMetrics')
    print(f'\nAvailable metrics: {matlab_cleaned_metrics.dtype.names}\n')
    try:
        metric_index = matlab_cleaned_metrics.dtype.names.index(which_metric)
        return matlab_cleaned_metrics[0][0][metric_index]

    except: 
        raise ValueError(f'*** {which_metric} does not exist. ***')
            
try:
    matlab_file_path1 = Path(f'{DRIVE}/{MAIN_FOLDER}/{compare1_folder}/RELAXProcessed/')
    matlab_file_path2 = Path(f'{DRIVE}/{MAIN_FOLDER}/{compare2_folder}/RELAXProcessed/')
    
    matlab_cleaned_metrics1 = extract_cleaned_metric_of_interest(matlab_file_path1, 'All_SER')
    matlab_cleaned_metrics2 = extract_cleaned_metric_of_interest(matlab_file_path2, 'All_SER')

except:
    print('Data not available.')


# =============================================================================
# compare peaks (PAC)
# =============================================================================

# eyes open: comp1 vs comp2
print(f'\n***** {compare1} vs {compare2} - eyes open *****\n')
data_folder_1 = COMPARE1_PATH
data_folder_2 = COMPARE2_PATH
wildcard1 = '*_eo_*.json'
wildcard2 = '*_eo_*.json'

x1_user_defined = compare1
x2_user_defined = compare2
title1_user_defined = 'eo'
title2_user_defined = 'eo'

x1 = 'x_user_defined'
x2 = 'x_user_defined'
y1 = 'peak_theta'
y2 = 'peak_gamma'
filename_split_colnames = ['id','session','timepoint','eyestatus']
pair_cols = ['id','session','timepoint','eyestatus']

ttests, means, stds, df_compare_1_eo, df_compare_2_eo = get_differences_between_the_two_data(data_folder_1, data_folder_2, 
                                                                                         wildcard1, wildcard2, x1, y1, x2, y2, 
                                                                                         filename_split_colnames,
                                                                                         x1_user_defined, x2_user_defined,
                                                                                         title1_user_defined, title2_user_defined,
                                                                                         pair_cols = pair_cols)


# eyes closed: comp1 vs comp2
print(f'\n***** {compare1} vs {compare2} - eyes closed *****\n')
drive = DRIVE
data_folder_1 = COMPARE1_PATH
data_folder_2 = COMPARE2_PATH
wildcard1 = '*_ec_*.json'
wildcard2 = '*_ec_*.json'

x1_user_defined = compare1
x2_user_defined = compare2
title1_user_defined = 'ec'
title2_user_defined = 'ec'

x1 = 'x_user_defined'
x2 = 'x_user_defined'
y1 = 'peak_theta'
y2 = 'peak_gamma'
filename_split_colnames = ['id','session','timepoint','eyestatus']
pair_cols = ['id','session','timepoint','eyestatus']


ttests, means, stds, df_compare_1_ec, df_compare_2_ec = get_differences_between_the_two_data(data_folder_1, data_folder_2, 
                                                                                         wildcard1, wildcard2, x1, y1, x2, y2, 
                                                                                         filename_split_colnames,
                                                                                         x1_user_defined, x2_user_defined,
                                                                                         title1_user_defined, title2_user_defined,
                                                                                         pair_cols = pair_cols)

#  comp1: eyes open vs eyes closed
print(f'\n***** eyes open vs eyes closed -{compare1} *****\n')
drive = DRIVE
data_folder_1 = COMPARE1_PATH
data_folder_2 = COMPARE1_PATH
wildcard1 = '*_eo_*.json'
wildcard2 = '*_ec_*.json'

x1_user_defined = ''
x2_user_defined = ''
title1_user_defined = compare1
title2_user_defined = compare1

x1 = 'eyestatus'
x2 = 'eyestatus'
y1 = 'peak_theta'
y2 = 'peak_gamma'
filename_split_colnames = ['id','session','timepoint','eyestatus']
pair_cols = ['id','session','timepoint']


ttests, means, stds, df_compare_1b_eo, df_compare_1b_ec = get_differences_between_the_two_data(data_folder_1, data_folder_2, 
                                                                                         wildcard1, wildcard2, x1, y1, x2, y2, 
                                                                                         filename_split_colnames,
                                                                                         x1_user_defined, x2_user_defined,
                                                                                         title1_user_defined, title2_user_defined,
                                                                                         pair_cols = pair_cols)



#  comp2: eyes open vs eyes closed
print(f'\n***** eyes open vs eyes closed -{compare2} *****\n')
drive = DRIVE
data_folder_1 = COMPARE2_PATH
data_folder_2 = COMPARE2_PATH
wildcard1 = '*_eo_*.json'
wildcard2 = '*_ec_*.json'

x1_user_defined = ''
x2_user_defined = ''
title1_user_defined = compare2
title2_user_defined = compare2

x1 = 'eyestatus'
x2 = 'eyestatus'
y1 = 'peak_theta'
y2 = 'peak_gamma'
filename_split_colnames = ['id','session','timepoint','eyestatus']
pair_cols = ['id','session','timepoint']


ttests, means, stds, df_compare_2b_eo, df_compare_2b_ec = get_differences_between_the_two_data(data_folder_1, data_folder_2, 
                                                                                         wildcard1, wildcard2, x1, y1, x2, y2, 
                                                                                         filename_split_colnames,
                                                                                         x1_user_defined, x2_user_defined,
                                                                                         title1_user_defined, title2_user_defined,
                                                                                         pair_cols = pair_cols)





# =============================================================================
# matlab
# =============================================================================

matlab_file_path = Path(f'{DRIVE}/{MAIN_FOLDER}/{which_matlab_comparison}/RELAXProcessed/Cleaned_data/Epoched/results_matlab/glmf')

matlab_eo = scipy.io.loadmat(matlab_file_path.joinpath('matlab_eo.mat')).get('peak_freqs')
df_matlab_eo = pd.DataFrame(matlab_eo, columns=['peak_theta', 'peak_gamma'])
df_matlab_eo[['id', 'session', 'timepoint']] = df_compare_1_eo[['id', 'session', 'timepoint']].copy()
df_matlab_eo['filtertype'] = 'glmf'
df_matlab_eo['eyestatus'] = 'eo'
df_matlab_eo['x_user_defined'] = 'glmf'

matlab_ec = scipy.io.loadmat(matlab_file_path.joinpath('matlab_ec.mat')).get('peak_freqs')
df_matlab_ec = pd.DataFrame(matlab_ec, columns=['peak_theta', 'peak_gamma'])
df_matlab_ec[['id', 'session', 'timepoint']] = df_compare_1_ec[['id', 'session', 'timepoint']].copy()
df_matlab_ec['filtertype'] = 'glmf'
df_matlab_ec['eyestatus'] = 'ec'
df_matlab_ec['x_user_defined'] = 'glmf'


ttests_matlab_eo_vs_ec, means_matlab_eo_vs_ec, stds_matlab_eo_vs_ec = test_for_difference(df_matlab_eo, df_matlab_ec, has_sig=False)



# =============================================================================
# #### matlab 1Hz_HighPass_MWF_01lostep_1bw vs 1Hz_HighPass_MWF_01lostep_2bw ##
# =============================================================================

# matlab_file_path = Path(f'{DRIVE}/EEG_data_collected_today/processed_EEG/1Hz_HighPass_MWF_01lostep_1bw/RELAXProcessed/Cleaned_data/Epoched/results_matlab/glmf')

# matlab_eo_1bw = scipy.io.loadmat(matlab_file_path.joinpath('matlab_eo.mat')).get('peak_freqs')
# df_matlab_eo_1bw = pd.DataFrame(matlab_eo_1bw, columns=['peak_theta', 'peak_gamma'])
# df_matlab_eo_1bw[['id', 'session', 'timepoint']] = df_compare_1_eo[['id', 'session', 'timepoint']].copy()
# df_matlab_eo_1bw['filtertype'] = 'glmf'
# df_matlab_eo_1bw['eyestatus'] = 'eo'
# df_matlab_eo_1bw['x_user_defined'] = 'glmf'

# matlab_ec_1bw = scipy.io.loadmat(matlab_file_path.joinpath('matlab_ec.mat')).get('peak_freqs')
# df_matlab_ec_1bw = pd.DataFrame(matlab_ec_1bw, columns=['peak_theta', 'peak_gamma'])
# df_matlab_ec_1bw[['id', 'session', 'timepoint']] = df_compare_1_ec[['id', 'session', 'timepoint']].copy()
# df_matlab_ec_1bw['filtertype'] = 'glmf'
# df_matlab_ec_1bw['eyestatus'] = 'ec'
# df_matlab_ec_1bw['x_user_defined'] = 'glmf'


# matlab_file_path = Path(f'{DRIVE}/EEG_data_collected_today/processed_EEG/1Hz_HighPass_MWF_01lostep_2bw/RELAXProcessed/Cleaned_data/Epoched/results_matlab/glmf')

# matlab_eo_2bw = scipy.io.loadmat(matlab_file_path.joinpath('matlab_eo.mat')).get('peak_freqs')
# df_matlab_eo_2bw = pd.DataFrame(matlab_eo_2bw, columns=['peak_theta', 'peak_gamma'])
# df_matlab_eo_2bw[['id', 'session', 'timepoint']] = df_compare_1_eo[['id', 'session', 'timepoint']].copy()
# df_matlab_eo_2bw['filtertype'] = 'glmf'
# df_matlab_eo_2bw['eyestatus'] = 'eo'
# df_matlab_eo_2bw['x_user_defined'] = 'glmf'

# matlab_ec_2bw = scipy.io.loadmat(matlab_file_path.joinpath('matlab_ec.mat')).get('peak_freqs')
# df_matlab_ec_2bw = pd.DataFrame(matlab_ec_2bw, columns=['peak_theta', 'peak_gamma'])
# df_matlab_ec_2bw[['id', 'session', 'timepoint']] = df_compare_1_ec[['id', 'session', 'timepoint']].copy()
# df_matlab_ec_2bw['filtertype'] = 'glmf'
# df_matlab_ec_2bw['eyestatus'] = 'ec'
# df_matlab_ec_2bw['x_user_defined'] = 'glmf'

# ttests_matlab_eo_1bw_vs_2bw, means_matlab_eo_1bw_vs_2bw, stds_matlab_eo_1bw_vs_2bw = test_for_difference(df_matlab_eo_1bw, df_matlab_eo_2bw, has_sig=False)
# ttests_matlab_ec_1bw_vs_2bw, means_matlab_ec_1bw_vs_2bw, stds_matlab_ec_1bw_vs_2bw = test_for_difference(df_matlab_ec_1bw, df_matlab_ec_2bw, has_sig=False)


# =============================================================================
# matlab vs python (theta and gamma separately)
# =============================================================================

# glmf (matlab) eo vs ec
print('\n***** eyes open vs eyes closed - MATLAB (glmf) *****\n')
df = pd.concat([df_matlab_eo, df_matlab_ec])
stripplot_comparison(x='eyestatus', y='peak_theta', df=df, title='glmf (matlab)', pair_cols = ['id','session','timepoint'])
stripplot_comparison(x='eyestatus', y='peak_gamma', df=df, title='glmf (matlab)', pair_cols = ['id','session','timepoint'])


# glmf (matlab) vs comp1 - eo
print(f'\n***** MATLAB vs {compare1} - eyes open *****\n')
df = pd.concat([df_matlab_eo, df_compare_1_eo])
stripplot_comparison(x='x_user_defined', y='peak_theta', df=df, title='eo', pair_cols = ['id','session','timepoint'])
stripplot_comparison(x='x_user_defined', y='peak_gamma', df=df, title='eo', pair_cols = ['id','session','timepoint'])
ttests_eo_matlab_vs_compare_1_eo, means_eo_matlab_vs_compare_1_eo, stds_eo_matlab_vs_compare_1_eo = test_for_difference(df_matlab_eo, df_compare_1_eo, has_sig=False)


# glmf (matlab) vs comp1 - ec
print(f'\n***** MATLAB vs {compare1} - eyes closed *****\n')
df = pd.concat([df_matlab_ec, df_compare_1_ec])
stripplot_comparison(x='x_user_defined', y='peak_theta', df=df, title='ec', pair_cols = ['id','session','timepoint'])
stripplot_comparison(x='x_user_defined', y='peak_gamma', df=df, title='ec', pair_cols = ['id','session','timepoint'])
ttests_ec_matlab_vs_compare_1_ec, means_ec_matlab_vs_compare_1_ec, stds_ec_matlab_vs_compare_1_ec = test_for_difference(df_matlab_ec, df_compare_1_ec, has_sig=False)


# glmf (matlab) vs comp2 - eo
print(f'\n***** MATLAB vs {compare2} - eyes open *****\n')
df = pd.concat([df_matlab_eo, df_compare_2_eo])
stripplot_comparison(x='x_user_defined', y='peak_theta', df=df, title='eo', pair_cols = ['id','session','timepoint'])
stripplot_comparison(x='x_user_defined', y='peak_gamma', df=df, title='eo', pair_cols = ['id','session','timepoint'])
ttests_eo_matlab_vs_compare_2_eo, means_eo_matlab_vs_compare_2_eo, stds_eo_matlab_vs_compare_2_eo = test_for_difference(df_matlab_eo, df_compare_2_eo, has_sig=False)


# glmf (matlab) vs comp2 - ec
print(f'\n***** MATLAB vs {compare2} - eyes closed *****\n')
df = pd.concat([df_matlab_ec, df_compare_2_ec])
stripplot_comparison(x='x_user_defined', y='peak_theta', df=df, title='ec', pair_cols = ['id','session','timepoint'])
stripplot_comparison(x='x_user_defined', y='peak_gamma', df=df, title='ec', pair_cols = ['id','session','timepoint'])
ttests_ec_matlab_vs_compare_2_ec, means_ec_matlab_vs_compare_2_ec, stds_ec_matlab_vs_compare_2_ec = test_for_difference(df_matlab_ec, df_compare_2_ec, has_sig=False)



# =============================================================================
# matlab vs python (distance)
# =============================================================================

# comp1 vs comp2 eo distance from glmf
print(f'\n***** {compare1} vs {compare2} - eyes open - distance from MATLAB (glmf) *****\n')
dist_matlab_df_compare_1_eo = get_distance(df_matlab_eo, df_compare_1_eo, 'peak_theta', 'peak_gamma')
dist_matlab_df_compare_2_eo = get_distance(df_matlab_eo, df_compare_2_eo, 'peak_theta', 'peak_gamma')
print('\n[Mean Distance (the lower the better)]')
print(f'matlab_to_{compare1}-eo: {round(dist_matlab_df_compare_1_eo.mean(), 2)}\nmatlab_to_{compare2}-eo: {round(dist_matlab_df_compare_2_eo.mean(), 2)}')
ttest_dist_eo = pingouin.ttest(dist_matlab_df_compare_1_eo, dist_matlab_df_compare_2_eo, paired=True)

df_glmf_and_df_compare_1_distance = pd.DataFrame(data=dist_matlab_df_compare_1_eo, columns=['distance'])
df_glmf_and_df_compare_1_distance['x_user_defined'] =  f'glmf_vs_{compare1}'
df_glmf_and_df_compare_1_distance[['id','session','timepoint','eyestatus']] = df_compare_1_eo[['id','session','timepoint','eyestatus']]

df_glmf_and_df_compare_2_distance = pd.DataFrame(data=dist_matlab_df_compare_2_eo, columns=['distance'])
df_glmf_and_df_compare_2_distance['x_user_defined'] = f'glmf_vs_{compare2}'
df_glmf_and_df_compare_2_distance[['id','session','timepoint','eyestatus']] = df_compare_2_eo[['id','session','timepoint','eyestatus']]


df = pd.concat([df_glmf_and_df_compare_1_distance, df_glmf_and_df_compare_2_distance])
stripplot_comparison(x='x_user_defined', y='distance', df=df, title='eo', pair_cols = ['id','session','timepoint'])


# comp1 vs comp2 ec distance from glmf
print(f'\n***** {compare1} vs {compare2} - eyes closed - distance from MATLAB (glmf) *****\n')
dist_matlab_df_compare_1_ec = get_distance(df_matlab_ec, df_compare_1_ec, 'peak_theta', 'peak_gamma')
dist_matlab_df_compare_2_ec = get_distance(df_matlab_ec, df_compare_2_ec, 'peak_theta', 'peak_gamma')
print('\n[Mean Distance (the lower the better)]')
print(f'matlab_to_{compare1}-ec: {round(dist_matlab_df_compare_1_ec.mean(), 2)}\nmatlab_to_{compare2}-ec: {round(dist_matlab_df_compare_2_ec.mean(), 2)}')
ttest_dist_ec = pingouin.ttest(dist_matlab_df_compare_1_ec, dist_matlab_df_compare_2_ec, paired=True)

df_glmf_and_df_compare_1_distance = pd.DataFrame(data=dist_matlab_df_compare_1_ec, columns=['distance'])
df_glmf_and_df_compare_1_distance['x_user_defined'] =  f'glmf_vs_{compare1}'
df_glmf_and_df_compare_1_distance[['id','session','timepoint','eyestatus']] = df_compare_1_ec[['id','session','timepoint','eyestatus']]

df_glmf_and_df_compare_2_distance = pd.DataFrame(data=dist_matlab_df_compare_2_ec, columns=['distance'])
df_glmf_and_df_compare_2_distance['x_user_defined'] = f'glmf_vs_{compare2}'
df_glmf_and_df_compare_2_distance[['id','session','timepoint','eyestatus']] = df_compare_2_ec[['id','session','timepoint','eyestatus']]


df = pd.concat([df_glmf_and_df_compare_1_distance, df_glmf_and_df_compare_2_distance])
stripplot_comparison(x='x_user_defined', y='distance', df=df, title='ec', pair_cols = ['id','session','timepoint'])



try:
    run_multiple_regression_for_epoch_number(df_compare_1_eo,
                                             df_compare_2_eo,
                                             df_compare1_epoch_number,
                                             df_compare2_epoch_number)
except:
    print('no epoch information found.')
    
