# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 23:09:03 2023

This is just a quick and dirty way of comparing two PAC from each other.
Not meant to be anything more than that.

@author: sungw
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import pingouin
import numpy as np
from statannot import add_stat_annotation
import statsmodels.api as sm

# from statannotations.Annotator import Annotator


    
def get_files_containing_str(data_folder: str, wildcard_str: str) -> list[Path]:
    data_file_path = data_folder.glob(wildcard_str)
    return [i for i in data_file_path if i.is_file()]
    

def get_file_id_list(data_folder: str, list_of_files: list[Path]) -> list[str]:
    return [str(i).split(str(f'{data_folder}\\'))[1].split('_ind_peaks.json')[0] for i in list_of_files]


def get_first_in_list_of_lists(list_of_lists) -> list[list[tuple]]:
    return [i[0] if i else [0, 0] for i in list_of_lists]


def get_first_in_dict(in_dict: dict[list[list[float, float]]]) -> list[list[float, float]]:
    '''
    get first in the peak and sig_pixel. If empty then turn into [0, 0]
    '''
    peaks = in_dict.get('peak')
    sig_pixels = in_dict.get('sig_pixel')
    
    first_peak = [0, 0] if not peaks else peaks[0]
    first_sig_pixel = [0, 0] if not sig_pixels else sig_pixels[0]
    
    return [first_peak, first_sig_pixel]
    

def get_df_of_first_peaks(data_folder: Path, 
                          wildcard_str: str, 
                          filename_split_colnames : list[str] = ['id','session','timepoint','eyestatus','filtertype'],
                          add_col: str = '',) -> pd.DataFrame:
    
    # import pdb; pdb.set_trace()
    input_files = get_files_containing_str(data_folder, wildcard_str)
    # input_files = get_files_containing_str(data_folder, '*.pkl')

    # read data of interest
    # import pdb; pdb.set_trace()
    # list_of_data = [pd.read_pickle(i) for i in input_files]
    # peak_data = [get_first_in_list_of_lists(i) for i in list_of_data]

    
    def read_json(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
        
    list_of_data = [read_json(i) for i in input_files]
    
    peak_data = [get_first_in_dict(i) for i in list_of_data]
    
    data_id = get_file_id_list(data_folder, input_files)
    # import pdb; pdb.set_trace()
    peak_df = [[data_id[i]] + peak_data[i] for i in range(len(peak_data))]
    peak_df = pd.DataFrame(peak_df, columns=['filename','peak','sig_pixel'])
    
    peak_df_temp = peak_df['filename'].str.split('_', expand=True)  
    peak_df_temp = peak_df_temp.drop(columns=peak_df_temp\
                                     .columns[(peak_df_temp.applymap(lambda s: s.lower() if type(s) == str else s).\
                                               isin(['resting', 'relax','epoched','downsampled'])).any()])
    if peak_df_temp.empty:
        print('Something must be wrong! No peak found?!')
    else:
        peak_df[filename_split_colnames] = peak_df_temp
    
    # sometimes x is not available in the data, so we add it here.
    if add_col:
        peak_df['x_user_defined'] = add_col
    return peak_df


# def get_xy_coordinate_diffs(peaks_a: list[tuple[float]], peaks_b: list[tuple[float]], peak_or_sigpixel: str = 'peak'):
    
#     assert peak_or_sigpixel in ['peak', 'sigpixel'], "'peak_or_sigpixel' needs to be either 'peak' or 'sigpixel'"
#     peak_idx = 0 if peak_or_sigpixel == 'peak' else 1
#     x_coord_diff = peaks_a[peak_idx][0] - peaks_b[peak_idx][0]
#     y_coord_diff = peaks_a[peak_idx][1] - peaks_b[peak_idx][1]
    
#     return x_coord_diff, y_coord_diff


def split_tuple_into_columns(df: pd.DataFrame, invar: str) -> pd.DataFrame:
    return pd.DataFrame(df[invar].tolist(), index=df.index)


def get_distance(df1: pd.DataFrame, df2: pd.DataFrame, x_coordinate: str, y_coordinate: str) -> np.ndarray:
    '''
    getting distance between xy coordinates of dataframe 1 and dataframe 2
    
    e.g) get_distance(df_wavelet_eo, df_hilbert_eo, 'peak_theta', 'peak_gamma')
    '''
    x_diff = df1[x_coordinate] - df2[x_coordinate]
    y_diff = df1[y_coordinate] - df2[y_coordinate]
    return np.hypot(x_diff, y_diff)


def sig_or_not(df: pd.DataFrame, invar: str) -> int:
    '''
    showing that the peak is significant (True) or not (False)
    '''
    return np.where(df[invar] != 0, True, False)


def df_wrapper(data_folder: Path, wildcard_str: str, filename_split_colnames: list[str], x_user_defined: str, has_sig: bool = True) -> pd.DataFrame:
    # import pdb; pdb.set_trace()
    df = get_df_of_first_peaks(data_folder, wildcard_str, filename_split_colnames, x_user_defined)
    df[['peak_theta', 'peak_gamma']] = split_tuple_into_columns(df, 'peak')
    if has_sig:
        df[['sig_pixel_theta', 'sig_pixel_gamma']] = split_tuple_into_columns(df, 'sig_pixel')
        df['significant_peak'] = sig_or_not(df, 'sig_pixel_theta')
    return df


def test_for_difference(df1: pd.DataFrame, df2: pd.DataFrame, has_sig=True):
    '''
    getting some ttests and looking at standard deviations between two datasets
    '''
    
    # =============================================================================
    # t-test
    # =============================================================================
    ttest_theta_peak = pingouin.ttest(df1['peak_theta'], df2['peak_theta'], paired=True)
    ttest_gamma_peak = pingouin.ttest(df1['peak_gamma'], df2['peak_gamma'], paired=True)

    ttest_theta_sig_pixel = []; ttest_gamma_sig_pixel = []
    if has_sig:
        ttest_theta_sig_pixel = pingouin.ttest(df1['sig_pixel_theta'], df2['sig_pixel_theta'], paired=True)
        ttest_gamma_sig_pixel = pingouin.ttest(df1['sig_pixel_gamma'], df2['sig_pixel_gamma'], paired=True)   

        ttest_dict = {'theta_peak_diff': ttest_theta_peak,
                     'gamma_peak_diff': ttest_gamma_peak,
                     'theta_sig_pixel_diff': ttest_theta_sig_pixel,
                     'gamma_sig_pixel_diff': ttest_gamma_sig_pixel}    
    else:
        ttest_dict = {'theta_peak_diff': ttest_theta_peak,
                     'gamma_peak_diff': ttest_gamma_peak}

    ttest_df = pd.concat(ttest_dict.values(), ignore_index=True)
    ttest_df.index = ttest_dict.keys()

    # =============================================================================
    # mean
    # =============================================================================
    ## by all
    # import pdb; pdb.set_trace()
    mean_all_df1 = df1.mean(numeric_only=True)
    mean_all_df2 = df2.mean(numeric_only=True)
    
    mean_dict = {'mean_all_df1': mean_all_df1,
                'mean_all_df2': mean_all_df2}
    
    mean_df = pd.DataFrame.from_dict(mean_dict)
    mean_df['mean_diff'] = mean_df['mean_all_df1'] - mean_df['mean_all_df2']

    # =============================================================================
    # Standard deviation
    # =============================================================================
    ## by all
    std_all_df1 = df1.std(numeric_only=True)
    std_all_df2 = df2.std(numeric_only=True)
    
    ## by id
    std_by_id_df1 = df1.groupby('id').std(numeric_only=True).mean()
    std_by_id_df2 = df2.groupby('id').std(numeric_only=True).mean()
    
    std_dict = {'std_all_df1': std_all_df1,
                'std_all_df2': std_all_df2,
                'std_by_id_df1': std_by_id_df1,
                'std_by_id_df2': std_by_id_df2}
    
    std_df = pd.DataFrame.from_dict(std_dict)
    
    return ttest_df, mean_df, std_df


def stripplot_comparison(x: str, y: str, df: pd.DataFrame, which_test: str = 't-test_paired', 
                         title: str ='', pair_cols: list[str] = ['id','session','timepoint','eyestatus']):
    '''
    plotting for significance, just between two paired sample as default
    '''
    # import pdb; pdb.set_trace()
    plt.figure()
    order = list(df[x].unique())
    p = sns.stripplot(x=x, y=y, data=df, size=4, color=".5", order=order)
    plt.xticks(rotation=45, ha="right")
    
    # plot the mean line
    ax = sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'b', 'ls': '-', 'lw': 2},
                medianprops={'color': 'r', 'ls': '-', 'lw': 2},
                whiskerprops={'visible': False},
                zorder=10,
                x=x,
                y=y,
                data=df,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=p,
                order=order)
    
    add_stat_annotation(ax, data=df, x=x, y=y, order=order,
                    box_pairs=[(order[0], order[1])],
                    test=which_test, text_format='simple', loc='inside', verbose=2)
    
    df['pair'] = df[pair_cols].apply(lambda x: '_'.join(x), axis=1)
    sns.lineplot(
        data=df.reset_index(drop=True), x=x, y=y, units='pair',
        color=".9", estimator=None
    )

    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    # return ax
    


def get_differences_between_the_two_data(path_to_json_files_1: Path, 
                                         path_to_json_files_2: Path, 
                                         wildcard_file1: str, 
                                         wildcard_file2: str, 
                                         x1: str,
                                         y1: str,
                                         x2: str,
                                         y2: str,
                                         filename_split_colnames: list[str],
                                         x1_user_defined: str = '',
                                         x2_user_defined: str = '',
                                         title1_user_defined: str = '',
                                         title2_user_defined: str = '',
                                         pair_cols: list[str] = ['id','session','timepoint','eyestatus'],
                                         which_test = 't-test_paired'):
    
    df1 = df_wrapper(path_to_json_files_1, wildcard_file1, filename_split_colnames, x1_user_defined)
    df2 = df_wrapper(path_to_json_files_2, wildcard_file2, filename_split_colnames, x2_user_defined)
 
    ttests, means, stds = test_for_difference(df1, df2)
    
    df = pd.concat([df1, df2])
    if (not title1_user_defined) & (not title2_user_defined):
        title1 = wildcard_file1.split('*')[1].split('_')[1]
        title2 = wildcard_file2.split('*')[1].split('_')[1]
    else:
        title1 = title1_user_defined
        title2 = title2_user_defined
    
    stripplot_comparison(x = x1, y = y1, df = df, which_test = which_test, title = title1, pair_cols = pair_cols)
    stripplot_comparison(x = x2, y = y2, df = df, which_test = which_test, title = title2, pair_cols = pair_cols)
    return ttests, means, stds, df1, df2


def get_epoch_number_df(compare_epoch_number_path):
    try:
        # with open(compare_epoch_number_path.joinpath('epoch_number.json'), 'r') as f:
        with open(list(compare_epoch_number_path.glob('epoch_*.json'))[0], 'r') as f:

            compare_epoch_number = json.load(f)
            df_compare_epoch_number = pd.DataFrame(compare_epoch_number).sort_values('filename').reset_index(drop=True)
            df_temp = df_compare_epoch_number['filename'].str.split('_', expand=True)  
            df_temp = df_temp.drop(columns=df_temp\
                                             .columns[~(df_temp.applymap(lambda s: s.lower() if type(s) == str else s).\
                                                       isin(['eo'])).any()])
                
            df_compare_epoch_number['eye_status'] = df_temp
            comp_epoch_mean_std = df_compare_epoch_number.drop(columns='filename').groupby('eye_status').agg(['mean','std'])
            
            df_compare_epoch_number_eo = df_compare_epoch_number.loc[df_compare_epoch_number['eye_status'] == 'eo', :]
            df_compare_epoch_number_ec = df_compare_epoch_number.loc[df_compare_epoch_number['eye_status'] == 'ec', :]

        return df_compare_epoch_number, comp_epoch_mean_std, df_compare_epoch_number_eo, df_compare_epoch_number_ec

    except:
        print('No file found, most likely')



def run_multiple_regression_for_epoch_number(df1, df2, df_epoch1, df_epoch2):
    df = pd.DataFrame()
    df['theta_a'] = df1['peak_theta']
    df['theta_b'] = df2['peak_theta']
    
    df['epoch_a'] = df_epoch1['epoch_number']
    df['epoch_b'] = df_epoch2['epoch_number']
    
    # calculate Theta Difference
    df['theta_diff'] = df['theta_a'] - df['theta_b']
    
    # create design matrix and response variable
    X = df[['epoch_a', 'epoch_b']]
    y = df['theta_diff']
    
    # add constant to design matrix
    X = sm.add_constant(X)
    
    # fit multiple linear regression model
    model = sm.OLS(y, X)
    results = model.fit()
    
    sns.pairplot(df[['theta_a', 'theta_b', 'epoch_a', 'epoch_b']])
    
    
    # Create a residual plot
    plt.figure()
    sns.residplot(x=df['theta_diff'], y=results.resid)
    plt.figure()
    sns.residplot(x='epoch_a', y='theta_diff', data=df, color='r')
    plt.figure()
    sns.residplot(x='epoch_b', y='theta_diff', data=df, color='g')
    


    # print summary of model
    print(results.summary())
    
