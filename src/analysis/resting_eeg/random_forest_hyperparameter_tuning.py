# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:19:03 2023

@author: sungw
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sns

SEED=9

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

def replace_zeros_and_nans_with_median(df, columns_to_process):
    # Calculate the median for each specified column (ignoring NaN values)
    median_values = df[columns_to_process].median()

    # Iterate through specified columns
    for column in columns_to_process:
        # Replace 0 and NaN values with the median of the column
        df[column] = df[column].replace({0: median_values[column], np.nan: median_values[column]})
    
    return df

SEED=9

drive = 'o'

df_main = pd.read_csv(Path(rf"{drive}:\CogTx\DataTransfer\ALZ TBS\z_alz_tbs_monash\id_cond_no_dropout_with_info.csv"))

## mtmfft
df_roi_bl = pd.read_csv(Path(rf"{drive}:\CogTx\DataTransfer\ALZ TBS\e. EEG data\Analysis2021\RELAX_Epoched_ALL\Power\results\ROI\ec_sham_BL_vs_ec_active_BL.csv"))
df_roi_delta = pd.read_csv(Path(rf"{drive}:\CogTx\DataTransfer\ALZ TBS\e. EEG data\Analysis2021\RELAX_Epoched_ALL\Power\results\ROI\ec_sham_delta_vs_ec_active_delta.csv"))

# =============================================================================
# theta gamma coupling data
# =============================================================================
df_tgc_median = pd.read_csv(Path(rf"{drive}:\CogTx\DataTransfer\ALZ TBS\e. EEG data\TGC\processed_EEG\025Hz_NoMWF_ALL\RELAXProcessed\Cleaned_Data\Epoched\results_pac\median\output_025Hz_NoMWF_ALL.csv"), index_col=[0]).rename(columns={'id':'ID'})

df_tgc_median_ec_bl = df_tgc_median[(df_tgc_median['eyestatus'] == 'eyesclosed') & (df_tgc_median['timepoint'] == 'BL')][['ID','peak_theta', 'peak_gamma']].rename(columns={'peak_theta':'peak_theta_ec','peak_gamma':'peak_gamma_ec'})
df_tgc_median_eo_bl = df_tgc_median[(df_tgc_median['eyestatus'] == 'eyesopen') & (df_tgc_median['timepoint'] == 'BL')][['ID','peak_theta', 'peak_gamma']].rename(columns={'peak_theta':'peak_theta_eo','peak_gamma':'peak_gamma_eo'})

out_dir = Path(rf"{drive}:\CogTx\DataTransfer\ALZ TBS\z_alz_tbs_monash\model_output")
out_dir.mkdir(parents=True, exist_ok=True)


df_roi_bl_long = turn_df_wide_to_long(df_roi_bl)
df_roi_delta_long = turn_df_wide_to_long(df_roi_delta)

## merging all the data together
# df = df_main.merge(df_roi_bl_long, on=['ID','COND'], how='inner').merge(df_roi_delta_long, on=['ID','COND'], how='inner')
df_combined = df_main.merge(df_roi_bl_long, on=['ID','COND'], how='inner')
df_combined = pd.merge(df_combined, df_tgc_median_ec_bl, how='left', on='ID')
df_combined = pd.merge(df_combined, df_tgc_median_eo_bl, how='left', on='ID')
df_combined = replace_zeros_and_nans_with_median(df_combined, ['peak_theta_ec', 'peak_gamma_ec', 'peak_theta_eo', 'peak_gamma_eo'])

df_combined['tgc_ec_ratio_tg'] = df_combined['peak_theta_ec'] / df_combined['peak_gamma_ec']
df_combined['tgc_eo_ratio_tg'] = df_combined['peak_theta_eo'] / df_combined['peak_gamma_eo']

df_combined['tgc_ec_ratio_gt'] = df_combined['peak_gamma_ec'] / df_combined['peak_theta_ec']
df_combined['tgc_eo_ratio_gt'] = df_combined['peak_gamma_eo'] / df_combined['peak_theta_eo']


# =============================================================================
# Other settings
# =============================================================================
## xgboost needs it to be 0 and 1
df_combined['Responder_1'].replace({2: 0}, inplace=True)
df_combined['Responder_2'].replace({2: 0}, inplace=True)


df = df_combined[df_combined['COND'].isin(['active'])]
X = df[['mean_gamma_connectivity_BL']]
y = df['Responder_2'].ravel()


# Set up the hyperparameter grid to search
param_grid = {
    'n_estimators': [10, 50, 100, 200, 300, 400, 500],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
    'class_weight': [None, 'balanced', 'balanced_subsample'],
    'max_samples': [None, 0.5, 0.75, 0.9]
}


# Make a custom scorer for accuracy
accuracy_scorer = make_scorer(accuracy_score)
min_class_size = min(np.bincount(y))
# Define the stratified K-fold cross-validator
n_splits = min(10, min_class_size)  # n_splits should not exceed the number of instances in the smallest class
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

# Make a custom scorer for accuracy
accuracy_scorer = make_scorer(accuracy_score)

# Create a base model
rf = RandomForestClassifier(random_state=SEED)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=skf, n_jobs=-1, scoring=accuracy_scorer, verbose=2)

# Fit the grid search to the data
grid_search.fit(X, y)

# Best model after grid search
best_rf = grid_search.best_estimator_

# Now, let's evaluate the best model from the grid search with cross-validation
# to plot ROC and PR curves, confusion matrix, and save metrics

# Initialize metrics
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
roc_auc_list = []
pr_auc_list = []
confusion_mat_list = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit the model and predict
    best_rf.fit(X_train_scaled, y_train)
    y_pred = best_rf.predict(X_test_scaled)
    y_proba = best_rf.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    accuracy_list.append(accuracy_score(y_test, y_pred))
    precision_list.append(precision_score(y_test, y_pred))
    recall_list.append(recall_score(y_test, y_pred))
    f1_list.append(f1_score(y_test, y_pred))
    roc_auc_list.append(roc_auc_score(y_test, y_proba))
    pr_auc_list.append(average_precision_score(y_test, y_proba))
    confusion_mat_list.append(confusion_matrix(y_test, y_pred))

# Calculating mean metric values
mean_accuracy = np.mean(accuracy_list)
mean_precision = np.mean(precision_list)
mean_recall = np.mean(recall_list)
mean_f1 = np.mean(f1_list)
mean_roc_auc = np.mean(roc_auc_list)
mean_pr_auc = np.mean(pr_auc_list)

# Aggregate confusion matrix
agg_confusion_mat = np.sum(confusion_mat_list, axis=0)

# Plotting Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(agg_confusion_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Aggregated Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plotting ROC Curve
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
for i in range(n_splits):
    fpr, tpr, _ = roc_curve(y[test_index], best_rf.predict_proba(X_test_scaled)[:, 1])
    plt.plot(fpr, tpr, label=f'Fold {i+1} (AUC = {roc_auc_list[i]:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Plotting PR Curve
plt.figure(figsize=(8, 6))
for i in range(n_splits):
    precision, recall, _ = precision_recall_curve(y[test_index], best_rf.predict_proba(X_test_scaled)[:, 1])
    plt.plot(recall, precision, label=f'Fold {i+1} (AP = {pr_auc_list[i]:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()


# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best parameters: {best_params}")
print(f"Best cross-validation accuracy: {best_score}")



# {'bootstrap': True,
#  'class_weight': None,
#  'criterion': 'gini',
#  'max_depth': None,
#  'max_features': 'sqrt',
#  'max_samples': 0.75,
#  'min_samples_leaf': 4,
#  'min_samples_split': 10,
#  'n_estimators': 200}

# Best cross-validation accuracy: 0.7333333333333333