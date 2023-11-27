# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:13:56 2023

@author: SChung
"""

from pathlib import Path
import pandas as pd
import statsmodels.api as sm

drive = 'o'

df_main = pd.read_csv(Path(rf"{drive}:\CogTx\DataTransfer\ALZ TBS\z_alz_tbs_monash\id_cond_no_dropout_with_info.csv"))

## mtmfft
df_roi_bl = pd.read_csv(Path(rf"{drive}:\CogTx\DataTransfer\ALZ TBS\e. EEG data\Analysis2021\RELAX_Epoched_ALL\Power\results\ROI\ec_sham_BL_vs_ec_active_BL.csv"))
df_roi_delta = pd.read_csv(Path(rf"{drive}:\CogTx\DataTransfer\ALZ TBS\e. EEG data\Analysis2021\RELAX_Epoched_ALL\Power\results\ROI\ec_sham_delta_vs_ec_active_delta.csv"))

## neil
# df_roi_bl = pd.read_csv(Path(r"O:\CogTx\DataTransfer\ALZ TBS\e. EEG data\Analysis2021\RELAX_Epoched_ALL\Power\results\neil\ROI\ec_sham_BL_vs_ec_active_BL.csv"))
# df_roi_delta = pd.read_csv(Path(r"O:\CogTx\DataTransfer\ALZ TBS\e. EEG data\Analysis2021\RELAX_Epoched_ALL\Power\results\neil\ROI\ec_sham_delta_vs_ec_active_delta.csv"))

def run_hierarchical_linear_regression(df, x_cols_list, y_col):
    models = []  # Initialize a list to store the models
    
    y = df[y_col]
    
    for x_cols in x_cols_list:
        X = df[x_cols]  # Select the columns for X variables
        X = sm.add_constant(X)  # Adds a constant term to the predictor
        
        model = sm.OLS(y, X).fit()
        models.append(model)  # Append the model to the list
        
        # Print model summary for each block
        print('\n-- Block Summary --')
        print(model.summary())
    
    return models


def turn_df_wide_to_long(df):
   # Separate dataframe based on 'sham' and 'active'
    df_sham = df.filter(regex='ID1|sham').copy()
    df_active = df.filter(regex='ID2|active').copy()
    
    # Rename columns by removing 'sham' or 'active' substring and rename ID columns
    df_sham.columns = df_sham.columns.str.replace('ec_sham_', 'ec_').str.replace('ID1', 'ID')
    df_active.columns = df_active.columns.str.replace('ec_active_', 'ec_').str.replace('ID2', 'ID')
    
    # Add 'COND' column to both dataframes
    df_sham['COND'] = 'sham'
    df_active['COND'] = 'active'
    
    # Concatenate the two dataframes
    df_long = pd.concat([df_sham, df_active], axis=0).sort_values(by='ID').reset_index(drop=True)
    df_long = df_long[df_long['ID'].notna()]
    return df_long


def run_regression(df, x_cols_list, y_col, active_or_sham):
    df_subset = df[df['COND'] == active_or_sham]
    
    print(f'\n\n\n<<< {active_or_sham} condition >>>')
    models = run_hierarchical_linear_regression(df_subset, x_cols_list, y_col)
    print()
    return models


df_roi_bl_long = turn_df_wide_to_long(df_roi_bl)
df_roi_delta_long = turn_df_wide_to_long(df_roi_delta)

## merging all the data together
df = df_main.merge(df_roi_bl_long, on=['ID','COND'], how='inner').merge(df_roi_delta_long, on=['ID','COND'], how='inner')


x_cols_list = [
    ## left frontal gamma causes multicolinearity
    
    # ['mean_gamma_connectivity_BL', 'Cogstate_list_recall_BSL'],    
    # ['mean_gamma_connectivity_BL', 'Cogstate_list_recall_BSL', 'change_mean_gamma_connectivity'],
    
    # ['mean_gamma_connectivity_BL', 'ec_BL LP gamma', 'ec_BL RF gamma', 'ec_BL RF theta', 'Cogstate_list_recall_BSL'],
    # ['mean_gamma_connectivity_BL', 'ec_BL LP gamma', 'ec_BL RF gamma', 'ec_BL RF theta', 'Cogstate_list_recall_BSL', 'change_mean_gamma_connectivity'],

    ['mean_gamma_connectivity_BL', 'ec_BL LP gamma', 'ec_BL RF gamma', 'ec_BL RP gamma', 'Cogstate_list_recall_BSL'],
    ['mean_gamma_connectivity_BL', 'ec_BL LP gamma', 'ec_BL RF gamma', 'ec_BL RP gamma', 'Cogstate_list_recall_BSL', 'change_mean_gamma_connectivity'],


    # ['mean_gamma_connectivity_BL', 'ec_BL LP gamma', 'ec_BL RF gamma', 'ec_BL RP gamma',  'ec_BL RF theta', 'Cogstate_list_recall_BSL'],
    # ['mean_gamma_connectivity_BL', 'ec_BL LP gamma', 'ec_BL RF gamma', 'ec_BL RP gamma',  'ec_BL RF theta', 'Cogstate_list_recall_BSL', 'change_mean_gamma_connectivity'],



    # ['mean_gamma_connectivity_BL', 'ec_BL RF theta', 'Cogstate_list_recall_BSL'],
    # ['mean_gamma_connectivity_BL', 'ec_BL RF theta', 'Cogstate_list_recall_BSL', 'change_mean_gamma_connectivity'],
    
    # ['ec_BL ALL theta', 'Cogstate_list_recall_BSL'],
    # ['ec_BL ALL theta', 'Cogstate_list_recall_BSL', 'change_mean_gamma_connectivity']
    
    # ['ec_BL ALL alpha', 'Cogstate_list_recall_BSL'],
    # ['ec_BL ALL alpha', 'Cogstate_list_recall_BSL', 'change_mean_gamma_connectivity']
    
    # ['ec_BL ALL beta', 'Cogstate_list_recall_BSL'],
    # ['ec_BL ALL beta', 'Cogstate_list_recall_BSL', 'change_mean_gamma_connectivity']
    
    # ['ec_BL ALL gamma', 'Cogstate_list_recall_BSL'],
    # ['ec_BL ALL gamma', 'Cogstate_list_recall_BSL', 'change_mean_gamma_connectivity']
    
    # ['ec_BL ALL broad', 'Cogstate_list_recall_BSL'],
    # ['ec_BL ALL broad', 'Cogstate_list_recall_BSL', 'change_mean_gamma_connectivity']
    
    # ['ec_BL LF theta', 'ec_BL RF theta', 'ec_BL LP theta', 'ec_BL RP theta', 'Cogstate_list_recall_BSL'],
    # ['ec_BL LF theta', 'ec_BL RF theta', 'ec_BL LP theta', 'ec_BL RP theta', 'Cogstate_list_recall_BSL', 'change_mean_gamma_connectivity']

    # ['ec_BL LF alpha', 'ec_BL RF alpha', 'ec_BL LP alpha', 'ec_BL RP alpha', 'Cogstate_list_recall_BSL'],
    # ['ec_BL LF alpha', 'ec_BL RF alpha', 'ec_BL LP alpha', 'ec_BL RP alpha', 'Cogstate_list_recall_BSL', 'change_mean_gamma_connectivity']

    # ['ec_BL LF beta', 'ec_BL RF beta', 'ec_BL LP beta', 'ec_BL RP beta', 'Cogstate_list_recall_BSL'],
    # ['ec_BL LF beta', 'ec_BL RF beta', 'ec_BL LP beta', 'ec_BL RP beta', 'Cogstate_list_recall_BSL', 'change_mean_gamma_connectivity']

    # ['ec_BL LF gamma', 'ec_BL RF gamma', 'ec_BL LP gamma', 'ec_BL RP gamma', 'Cogstate_list_recall_BSL'],
    # ['ec_BL LF gamma', 'ec_BL RF gamma', 'ec_BL LP gamma', 'ec_BL RP gamma', 'Cogstate_list_recall_BSL', 'change_mean_gamma_connectivity']
    
    # ['ec_BL LF broad', 'ec_BL RF broad', 'ec_BL LP broad', 'ec_BL RP broad', 'Cogstate_list_recall_BSL'],
    # ['ec_BL LF broad', 'ec_BL RF broad', 'ec_BL LP broad', 'ec_BL RP broad', 'Cogstate_list_recall_BSL', 'change_mean_gamma_connectivity'] 
    
    ]

y_col = ['change_list_recall']



def main():
    run_regression(df, x_cols_list, y_col, 'active')
    run_regression(df, x_cols_list, y_col, 'sham')



if __name__ == "__main__":
    main()