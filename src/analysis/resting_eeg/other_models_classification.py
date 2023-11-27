# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 17:53:27 2023

@author: sungw
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, recall_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import LeaveOneOut
import pickle
# from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB




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


conditions = [
            ['active'], 
            ['sham'],
            ['active', 'sham'],
              ]
target_variables = ['Responder_1', 'Responder_2']
fn_suffixes = ['gc', 'gc_gpow', 'gc_gpow_tgc', 'gc_allpow', 'gc_tgc', 'gc_allpow_tgc']


x_col_lists = [
    
                ['mean_gamma_connectivity_BL'],
                
                ['mean_gamma_connectivity_BL',
                'ec_BL LF gamma',
                'ec_BL RF gamma',
                'ec_BL LP gamma',
                'ec_BL RP gamma',],
                
                ['mean_gamma_connectivity_BL',
                'ec_BL LF gamma',
                'ec_BL RF gamma',
                'ec_BL LP gamma',
                'ec_BL RP gamma',
                
                'peak_theta_ec',
                'peak_gamma_ec',
                'tgc_ec_ratio_tg',
                'tgc_ec_ratio_gt',],
                
                
                
                ['mean_gamma_connectivity_BL',
                'ec_BL LF gamma',
                'ec_BL RF gamma',
                'ec_BL LP gamma',
                'ec_BL RP gamma',
                
                'ec_BL LF theta',
                'ec_BL RF theta',
                'ec_BL LP theta',
                'ec_BL RP theta',
                
                'ec_BL LF alpha',
                'ec_BL RF alpha',
                'ec_BL LP alpha',
                'ec_BL RP alpha',
                
                'ec_BL LF beta',
                'ec_BL RF beta',
                'ec_BL LP beta',
                'ec_BL RP beta',],
                
                ['mean_gamma_connectivity_BL',
                'peak_theta_ec',
                'peak_gamma_ec',
                'tgc_ec_ratio_tg',
                'tgc_ec_ratio_gt',],
                
                ['mean_gamma_connectivity_BL',
                'ec_BL LF gamma',
                'ec_BL RF gamma',
                'ec_BL LP gamma',
                'ec_BL RP gamma',
                
                'ec_BL LF theta',
                'ec_BL RF theta',
                'ec_BL LP theta',
                'ec_BL RP theta',
                
                'ec_BL LF alpha',
                'ec_BL RF alpha',
                'ec_BL LP alpha',
                'ec_BL RP alpha',
                
                'ec_BL LF beta',
                'ec_BL RF beta',
                'ec_BL LP beta',
                'ec_BL RP beta',
                'peak_theta_ec',
                'peak_gamma_ec',
                'tgc_ec_ratio_tg',
                'tgc_ec_ratio_gt',],
            
                ]



for condition in conditions:
    for target_variable in target_variables:
        for fn_suffix, x_col_list in zip(fn_suffixes, x_col_lists):



            df = df_combined[df_combined['COND'].isin(condition)]
            
            concatenated_condition = "-".join(condition)
            
            subfolder = f'{target_variable}_{concatenated_condition}_{fn_suffix}'
            subfolder_dir = out_dir.joinpath(subfolder)
            subfolder_dir.mkdir(parents=True, exist_ok=True)
            
            average_method = 'weighted'
            
            n_bootstraps = 1000  # Number of bootstrap samples
            n_splits = 5 # stratified kfold
            
            
            included_base_models = [
                                    'Random Forest', 
                                    'Gradient Boosting', 
                                    'SVM', 
                                    'Logistic Regression', 
                                    'XGBoost',
                                    'MLPClassifier',
                                    'KNeighborsClassifier',
                                    'AdaBoostClassifier',
                                    'GaussianNB'
                                    ]
            

            # Split the data
            X = df[x_col_list]
            y = df[target_variable]  # Target variable
            
            # Correlation-based feature selection (using entire dataset for finding correlations)
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            X_reduced = X.drop(to_drop, axis=1)
            
            X_reduced.columns.to_list()
            
            base_estimator = DecisionTreeClassifier(max_depth=1)
            
            # Define the base models
            base_model_candidates = {
                'Random Forest': RandomForestClassifier(
                                    n_estimators=200,
                                    max_depth=None,  # None means nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
                                    min_samples_split=2,
                                    min_samples_leaf=1,
                                    max_features='sqrt',  # 'auto' uses all features; try 'sqrt' or 'log2' for large feature sets
                                    bootstrap=True,  # Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
                                    random_state=SEED
                                ),
                'Gradient Boosting': GradientBoostingClassifier(
                                    n_estimators=200,
                                    learning_rate=0.1,  # Smaller values require more trees but can lead to better performance.
                                    max_depth=3,  # Control the depth of the individual regression estimators.
                                    min_samples_split=2,
                                    min_samples_leaf=1,
                                    subsample=1.0,  # Use a fraction of the data for fitting individual base learners.
                                    random_state=SEED
                                ),
                'SVM': SVC(
                            C=1.0,  # Regularization parameter
                            kernel='rbf',  # Try other kernels like 'linear', 'poly', and 'sigmoid'.
                            degree=3,  # Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.
                            gamma='scale',  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. If 'auto', uses 1 / n_features.
                            probability=True,
                            random_state=SEED
                        ),
                'Logistic Regression': LogisticRegression(
                                        penalty='l2',  # Try 'l1' for Lasso regularization.
                                        C=1.0,  # Inverse of regularization strength; smaller values specify stronger regularization.
                                        solver='lbfgs',  # For small datasets, 'liblinear' might be a good choice. For larger, 'saga' is good for both l1 and l2.
                                        max_iter=100,
                                        random_state=SEED
                                    ),
                
                'XGBoost': XGBClassifier(
                                    n_estimators=200,
                                    learning_rate=0.1,
                                    max_depth=3,  # Increasing this value will make the model more complex and likely to overfit.
                                    min_child_weight=1,  # Minimum sum of instance weight (hessian) needed in a child.
                                    subsample=1,  # Subsample ratio of the training instances.
                                    colsample_bytree=1,  # Subsample ratio of columns when constructing each tree.
                                    eval_metric='logloss',
                                    random_state=SEED
                                ),
                'MLPClassifier': 
                    MLPClassifier(
                                    hidden_layer_sizes=(100,), # You can try different configurations, e.g., (50,50)
                                    activation='relu',         # or 'tanh' if 'relu' doesn't work
                                    solver='adam',             # or 'lbfgs' for small datasets
                                    alpha=0.0001,              # Regularization term, adjust as necessary
                                    batch_size='auto',         # 'auto' sets it to min(200, n_samples)
                                    learning_rate='constant',  # or 'adaptive' if 'constant' doesn't work
                                    learning_rate_init=0.001,  # Start with default and adjust if necessary
                                    max_iter=3000,             # You've already tried up to this
                                    random_state=SEED,
                                    early_stopping=True,      # Set to True to use automatic stopping
                                    validation_fraction=0.1,   # Fraction of the data for validation (used when early_stopping is True)
                                    n_iter_no_change=10
                                    ),
                'KNeighborsClassifier': KNeighborsClassifier(
                                    n_neighbors=5,  # Number of neighbors to use.
                                    weights='uniform',  # 'distance' weights points by the inverse of their distance.
                                    algorithm='auto',  # 'ball_tree', 'kd_tree', and 'brute' can also be chosen based on dataset size.
                                    leaf_size=30,  # Leaf size passed to BallTree or KDTree.
                                    p=2  # Power parameter for the Minkowski metric.
                                ),
                'AdaBoostClassifier': AdaBoostClassifier(
                                    estimator=base_estimator,  
                                    n_estimators=50,
                                    learning_rate=1.0,
                                    random_state=SEED
                                ),
                
                "GaussianNB": GaussianNB(),
                    }
            
            
            base_models = {model_name: base_model_candidates[model_name] for model_name in included_base_models}
            
            # =============================================================================
            # BOOTSTRAP StratifiedShuffle
            # =============================================================================
            
            bootstrap_results = {}
            
            # Create a stratified split to ensure each sample maintains class proportions
            sss = StratifiedShuffleSplit(n_splits=n_bootstraps, test_size=0.5, random_state=SEED)
            smote = SMOTE(random_state=SEED)  # Instantiate SMOTE
            
            roc_curves = {name: {'fpr': [], 'tpr': [], 'auc': []} for name in base_models.keys()}
            
            # Additional dictionaries to store the precision, recall for each model
            pr_curves = {name: {'precision': [], 'recall': [], 'pr_auc': []} for name in base_models.keys()}
            
            # Store the sum of confusion matrices for each model to calculate the mean later
            confusion_matrix_sum = {name: np.zeros((2, 2)) for name in base_models.keys()}
            
            # Initialize a dictionary to store the feature importances
            feature_importances = {name: [] for name in base_models.keys()}
            
            for name, model in base_models.items():
                accuracies = []
                f1_scores = []
                auc_scores = []
                pr_auc_scores = []
                precision_scores = []
                recall_scores = []
                confusion_matrices = []
                
                # Initialize a list to hold the feature importances for each bootstrap iteration
                bootstrap_feature_importances = []
            
                for train_index, test_index in sss.split(X_reduced, y):
                    # Bootstrap samples
                    X_train, y_train = X_reduced.iloc[train_index], y.iloc[train_index]
                    X_test, y_test = X_reduced.iloc[test_index], y.iloc[test_index]
            
                    # Check if SMOTE should be applied
                    if y_train.value_counts().min() > 1:  # More than one sample in the minority class
                        # Adjust `k_neighbors` parameter if needed
                        k_neighbors = min(y_train.value_counts().min() - 1, smote.k_neighbors)
                        smote.k_neighbors = k_neighbors  # or `smote = SMOTE(k_neighbors=k_neighbors, random_state=SEED)`
            
                        # Apply SMOTE
                        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
                    else:
                        # Not enough samples to apply SMOTE. Proceed with the original data
                        X_train_smote, y_train_smote = X_train, y_train
            
                    # Fit the model on the over-sampled dataset
                    model.fit(X_train_smote, y_train_smote)
                    
                    # Predict on the original test set
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
                    
                    # Calculate metrics
                    accuracies.append(accuracy_score(y_test, y_pred))
                    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
                    auc_scores.append(roc_auc_score(y_test, y_proba))
                    pr_auc_scores.append(average_precision_score(y_test, y_proba))
                    precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                    recall_scores.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
                    confusion_matrices.append(confusion_matrix(y_test, y_pred))
                    
                    # Calculate the fpr, tpr, and thresholds for each bootstrap
                    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    # Store the fpr, tpr, and auc
                    roc_curves[name]['fpr'].append(fpr)
                    roc_curves[name]['tpr'].append(tpr)
                    roc_curves[name]['auc'].append(roc_auc)
                    
                    
                    # Calculate precision, recall for the PR curve
                    precision, recall, _ = precision_recall_curve(y_test, y_proba)
                    pr_auc = average_precision_score(y_test, y_proba)
                    
                    # Store precision, recall, and pr_auc
                    pr_curves[name]['precision'].append(precision)
                    pr_curves[name]['recall'].append(recall)
                    pr_curves[name]['pr_auc'].append(pr_auc)
                    
                    # Sum the confusion matrix for later averaging
                    confusion_matrix_sum[name] += confusion_matrix(y_test, y_pred)
                    
                    # Check if the model is a tree-based model with feature importances
                    if hasattr(model, 'feature_importances_'):
                        # Append the feature importances for this bootstrap iteration
                        bootstrap_feature_importances.append(model.feature_importances_)
                
                    # If feature importances were collected, average them across all bootstrap iterations
                if bootstrap_feature_importances:
                    # Calculate the average feature importance for each feature
                    averaged_importances = np.mean(bootstrap_feature_importances, axis=0)
                    feature_importances[name] = averaged_importances
            
             
            
                bootstrap_results[name] = {
                    'Mean Accuracy': np.mean(accuracies),
                    'Mean F1 Score': np.mean(f1_scores),
                    'Mean ROC AUC': np.mean(auc_scores),
                    'Mean PR AUC': np.mean(pr_auc_scores),
                    'Mean Precision': np.mean(precision_scores),
                    'Mean Recall': np.mean(recall_scores),
                    'Confusion Matrices': confusion_matrices
                }
            
            
            # Display the results
            for name, metrics in bootstrap_results.items():
                print(f"---{name}---")
                print(f"Mean Accuracy: {metrics['Mean Accuracy']:.4f}")
                print(f"Mean F1 Score: {metrics['Mean F1 Score']:.4f}")
                print(f"Mean ROC AUC: {metrics['Mean ROC AUC']:.4f}")
                print(f"Mean PR AUC: {metrics['Mean PR AUC']:.4f}")
                print(f"Mean Precision: {metrics['Mean Precision']:.4f}")
                print(f"Mean Recall: {metrics['Mean Recall']:.4f}")
                # For confusion matrices, you may want to average them or just inspect individually
                print("\n")
                
            
            for name, data in roc_curves.items():

                # After your existing loop that collects metrics and ROC/PR data
                # Extracting FPR, TPR, and Thresholds from your roc_curves for Random Forest
                fpr_list = roc_curves[name]['fpr']
                tpr_list = roc_curves[name]['tpr']
                thresholds_list = [np.linspace(0, 1, len(fpr)) for fpr in fpr_list]  # Assume linearly spaced thresholds
                
                # Flattening the lists of FPRs, TPRs, and thresholds
                all_fpr = np.concatenate(fpr_list)
                all_tpr = np.concatenate(tpr_list)
                all_thresholds = np.concatenate(thresholds_list)
                
                # Sorting by descending thresholds
                sorted_indices = np.argsort(-all_thresholds)
                all_fpr_sorted = all_fpr[sorted_indices]
                all_tpr_sorted = all_tpr[sorted_indices]
                all_thresholds_sorted = all_thresholds[sorted_indices]
                
                # Calculating additional metrics for each threshold
                # Assuming you have a known prevalence
                prevalence = 0.5  # Example value for the proportion of positive cases in your dataset
                
                # Calculating additional metrics for each threshold
                specificity = 1 - all_fpr_sorted  # True Negative Rate
                plus_lr = np.divide(all_tpr_sorted, specificity, out=np.zeros_like(all_tpr_sorted), where=specificity!=0)
                minus_lr = np.divide(1 - all_tpr_sorted, specificity, out=np.zeros_like(all_tpr_sorted), where=specificity!=0)
                
                # Calculating predictive values
                plus_pv = (plus_lr * prevalence) / ((plus_lr * prevalence) + (1 - prevalence))
                minus_pv = (minus_lr * (1 - prevalence)) / ((minus_lr * (1 - prevalence)) + prevalence)
                
                # Handle edge cases for predictive values
                plus_pv[specificity == 0] = 1  # If specificity is 0, then +PV is 1
                minus_pv[all_tpr_sorted == 1] = 1  # If sensitivity is 1, then -PV is 1
                
                # Cost can be calculated based on your criteria or set to a default value
                # cost = np.zeros_like(all_thresholds_sorted)  # Example placeholder
                
                # Creating the criterion table DataFrame
                criterion_table = pd.DataFrame({
                    'Criterion': all_thresholds_sorted,
                    'Sensitivity': all_tpr_sorted,
                    'Specificity': specificity,
                    'P_LR': plus_lr,
                    'N_LR': minus_lr,
                    'P_PV': plus_pv,
                    'N_PV': minus_pv,
                    # 'Cost': cost
                })
                
                # Replace infinite values with NaN or a large number
                criterion_table.replace([np.inf, -np.inf], np.nan, inplace=True)
                
                # Fill NaN values if necessary
                criterion_table.fillna(value={'Cost': 0.0, 'P_LR': 999, 'P_LR': 999}, inplace=True)
                
                # Display the criterion table
                # print(criterion_table)
                
                # Assuming 'criterion_table' is your DataFrame
                criterion_table = criterion_table.drop_duplicates(subset='Criterion', keep='first').reset_index(drop=True)
                criterion_table = criterion_table.sort_values(by='Criterion', ascending=False).reset_index(drop=True)
                criterion_table.to_csv(subfolder_dir.joinpath(f'criterion_table_{target_variable}_{concatenated_condition}_{fn_suffix}_{name}.csv'), index=False)


            
            # After collecting all data, now plot the averaged feature importances for each tree-based model
            for name, importances in feature_importances.items():
                if len(importances) > 0:
                    # Sort the feature importances
                    indices = np.argsort(importances)[::-1]
                    sorted_importances = importances[indices]
                    sorted_features = X_reduced.columns[indices]
            
                    # Plot the feature importances
                    plt.figure(figsize=(10, 6))
                    plt.title(f'Average Feature Importances in {name}')
                    plt.bar(range(X_reduced.shape[1]), sorted_importances, color="r", align="center")
                    plt.xticks(range(X_reduced.shape[1]), sorted_features, rotation=90)
                    plt.xlim([-1, X_reduced.shape[1]])
                    plt.tight_layout()
                    plt.show()
                    plt.savefig(subfolder_dir.joinpath(f'feature_importance_Bootstrap_{target_variable}_{concatenated_condition}_{fn_suffix}_{name}.png'))
                    plt.close()

            
            
            
            
            
            # Plot the ROC curves
            plt.figure(figsize=(10, 8))
            for name, data in roc_curves.items():
                # Calculate mean fpr, tpr, and auc
                mean_fpr = np.linspace(0, 1, 100)
                mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(data['fpr'], data['tpr'])], axis=0)
                mean_auc = np.mean(data['auc'])
            
                plt.plot(mean_fpr, mean_tpr, label=f'{name} (AUC = {mean_auc:.2f})')
            
            
            
            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve per Model')
            plt.legend(loc="lower right")
            plt.show()
            plt.savefig(subfolder_dir.joinpath(f'ROC_curve_Bootstrap_{target_variable}_{concatenated_condition}_{fn_suffix}.png'))
            plt.close()

            
            
            
            # Plot the PR curves
            plt.figure(figsize=(10, 8))
            for name, data in pr_curves.items():
                # Calculate mean precision and recall
                mean_recall = np.linspace(0, 1, 100)
                mean_precision = np.mean([np.interp(mean_recall, recall[::-1], precision[::-1]) 
                                          for precision, recall in zip(data['precision'], data['recall'])], axis=0)
                mean_pr_auc = np.mean(data['pr_auc'])
            
                plt.plot(mean_recall, mean_precision, label=f'{name} (AP = {mean_pr_auc:.2f})')
            
            # Set x-axis and y-axis limits to [0, 1]
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve per Model')
            plt.legend(loc="lower left")
            plt.show()
            plt.savefig(subfolder_dir.joinpath(f'PR_curve_Bootstrap_{target_variable}_{concatenated_condition}_{fn_suffix}.png'))
            plt.close()

            
            
            # Plot the confusion matrices
            for name, matrix_sum in confusion_matrix_sum.items():
                # plt.figure()
                matrix_avg = matrix_sum / n_bootstraps
                disp = ConfusionMatrixDisplay(confusion_matrix=matrix_avg)
                disp.plot(cmap=plt.cm.Blues)
                plt.title(f'Average Confusion Matrix: {name}')
                plt.show()
                plt.savefig(subfolder_dir.joinpath(f'Confusion_Matrix_Bootstrap_{target_variable}_{concatenated_condition}_{fn_suffix}_{name}.png'))
                plt.close()

            

            # Save the dictionary using pickle
            with open(subfolder_dir.joinpath(f'bootstrap_results_{target_variable}_{concatenated_condition}_{fn_suffix}.pkl'), 'wb') as outfile:
                pickle.dump(bootstrap_results, outfile)
                
            df_bootstrap = pd.DataFrame.from_dict(bootstrap_results)
            df_bootstrap = df_bootstrap[~df_bootstrap.index.str.contains('Confusion')].T
            df_bootstrap.to_csv(subfolder_dir.joinpath(f'bootstrap_results_{target_variable}_{concatenated_condition}_{fn_suffix}.csv'))
            
            
                
            
            
                
            # =============================================================================
            # LEAVE ONE OUT CROSS VALIDATION
            # =============================================================================
                
            loo_results = {}
            
            # Create LOOCV object
            loo = LeaveOneOut()
            
            for name, model in base_models.items():
                accuracies = []
                f1_scores = []
                # auc_scores = []
                # pr_auc_scores = []
                precision_scores = []
                recall_scores = []
                confusion_matrices = []
            
                # Iterate over each train-test split
                for train_index, test_index in loo.split(X_reduced):
                                        
                    # Split the data
                    X_train, X_test = X_reduced.iloc[train_index], X_reduced.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                                
                    # Fit the model
                    model.fit(X_train, y_train)
            
                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0] * len(y_test)
            
                    # Calculate metrics
                    accuracies.append(accuracy_score(y_test, y_pred))
                    f1_scores.append(f1_score(y_test, y_pred, average=average_method))
                    # auc_scores.append(roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else 0.5)  # ROC AUC needs both classes
                    # pr_auc_scores.append(average_precision_score(y_test, y_proba) if len(set(y_test)) > 1 else 0.5)
                    precision_scores.append(precision_score(y_test, y_pred, average=average_method, zero_division=0))
                    recall_scores.append(recall_score(y_test, y_pred, average=average_method, zero_division=0))
                    confusion_matrices.append(confusion_matrix(y_test, y_pred))
            
                loo_results[name] = {
                    'Mean Accuracy': np.mean(accuracies),
                    'Mean F1 Score': np.mean(f1_scores),
                    # 'Mean ROC AUC': np.mean(auc_scores),
                    # 'Mean PR AUC': np.mean(pr_auc_scores),
                    'Mean Precision': np.mean(precision_scores),
                    'Mean Recall': np.mean(recall_scores),
                }
            
            # Display the results
            for name, metrics in loo_results.items():
                print(f"---{name}---")
                print(f"Mean Accuracy: {metrics['Mean Accuracy']:.4f}")
                print(f"Mean F1 Score: {metrics['Mean F1 Score']:.4f}")
                # print(f"Mean ROC AUC: {metrics['Mean ROC AUC']:.4f}")
                # print(f"Mean PR AUC: {metrics['Mean PR AUC']:.4f}")
                print(f"Mean Precision: {metrics['Mean Precision']:.4f}")
                print(f"Mean Recall: {metrics['Mean Recall']:.4f}")
                print("\n")
                
               
            # Save the dictionary using pickle
            with open(subfolder_dir.joinpath(f'loocv_results_{target_variable}_{concatenated_condition}_{fn_suffix}.pkl'), 'wb') as outfile:
                pickle.dump(loo_results, outfile)
                    
            df_loo = pd.DataFrame.from_dict(loo_results)
            df_loo = df_loo[~df_loo.index.str.contains('Confusion')].T
            df_loo.to_csv(subfolder_dir.joinpath(f'loocv_results_{target_variable}_{concatenated_condition}_{fn_suffix}.csv'))
                       
            
            # =============================================================================
            # Stratified KFold Validation
            # =============================================================================
            
            # Define the cross-validator
            # Adjust the number of splits based on the smallest class size
            min_class_size = min(y.value_counts())
            n_splits = min(n_splits, min_class_size)
            
            # Update the cross-validator
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            
            # Initialize dictionaries to store metrics
            model_metrics = {
                'Accuracy': {},
                'Recall': {},
                'F1 Score': {}
            }
            
            # Additional dictionaries to store the ROC and PR curve data
            roc_data = {name: {'tpr': [], 'fpr': [], 'thresholds': [], 'auc': []} for name in base_models.keys()}
            pr_data = {name: {'precision': [], 'recall': [], 'thresholds': [], 'pr_auc': []} for name in base_models.keys()}
            
            
            # Initialize dictionaries to store confusion matrix sums
            conf_matrix_sum = {name: np.zeros((2, 2)) for name in base_models.keys()}
            
            # Train and evaluate each model using cross-validation
            for name, model in base_models.items():
                accuracies = []
                recalls = []
                f1_scores = []
                
                # Lists to store each fold's ROC and PR data
                tprs = []
                fprs = []
                aucs = []
                precisions = []
                recalls_pr = []
                pr_aucs = []
                
                # Modify the model initialisation to account for class imbalance
                if "class_weight" in model.get_params().keys():
                    model.set_params(class_weight='balanced')
                
                # Split the data into K Folds
                for train_index, test_index in skf.split(X_reduced, y):
                    X_train_fold, X_test_fold = X_reduced.iloc[train_index], X_reduced.iloc[test_index]
                    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
                    
                    # Standardize features
                    scaler = StandardScaler()
                    X_train_fold = scaler.fit_transform(X_train_fold)
                    X_test_fold = scaler.transform(X_test_fold)
                    
                    # Fit the model
                    model.fit(X_train_fold, y_train_fold)
                    
                    # Make predictions
                    y_pred_fold = model.predict(X_test_fold)
                    y_pred_proba_fold = model.predict_proba(X_test_fold)[:, 1]
                    
                    # Skip AUC calculation if only one class is present in y_true
                    if len(np.unique(y_test_fold)) < 2:
                        # Log a message or handle as appropriate for your use case
                        print(f"Skipping AUC calculation for {name} on fold with insufficient class representation.")
                        continue
                    
                    # Compute metrics
                    accuracies.append(accuracy_score(y_test_fold, y_pred_fold))
                    recalls.append(recall_score(y_test_fold, y_pred_fold, average=average_method))
                    f1_scores.append(f1_score(y_test_fold, y_pred_fold, average=average_method))
                    
                    # Compute ROC data
                    fpr, tpr, roc_thresholds = roc_curve(y_test_fold, y_pred_proba_fold)
                    roc_auc = roc_auc_score(y_test_fold, y_pred_proba_fold)
                    fprs.append(fpr)
                    tprs.append(tpr)
                    aucs.append(roc_auc)
                    
                    # Compute PR data
                    precision, recall, pr_thresholds = precision_recall_curve(y_test_fold, y_pred_proba_fold)
                    pr_auc = average_precision_score(y_test_fold, y_pred_proba_fold)
                    precisions.append(precision)
                    recalls_pr.append(recall)
                    pr_aucs.append(pr_auc)
                    
                    # Compute confusion matrix and sum across folds
                    conf_matrix_sum[name] += confusion_matrix(y_test_fold, y_pred_fold)
                
                # Store the mean metrics
                model_metrics['Accuracy'][name] = np.mean(accuracies)
                model_metrics['Recall'][name] = np.mean(recalls)
                model_metrics['F1 Score'][name] = np.mean(f1_scores)
                
                # Store aggregated ROC and PR data
                roc_data[name]['tpr'] = tprs
                roc_data[name]['fpr'] = fprs
                roc_data[name]['auc'] = np.mean(aucs)
                
                pr_data[name]['precision'] = precisions
                pr_data[name]['recall'] = recalls_pr
                pr_data[name]['pr_auc'] = np.mean(pr_aucs)
            
            
            # =============================================================================
            # save csv
            # =============================================================================
            model_metrics_df = pd.DataFrame(model_metrics)
            model_metrics_df.to_csv(subfolder_dir.joinpath(f'model_metrics_summary_kfold_{target_variable}_{concatenated_condition}_{fn_suffix}.csv'))
            
            roc_auc_summary = {name: data['auc'] for name, data in roc_data.items()}
            roc_auc_df = pd.DataFrame.from_dict(roc_auc_summary, orient='index', columns=['Mean ROC AUC'])
            model_metrics_df.to_csv(subfolder_dir.joinpath(f'roc_auc_kfold_{target_variable}_{concatenated_condition}_{fn_suffix}.csv'))
            
            pr_auc_summary = {name: data['pr_auc'] for name, data in pr_data.items()}
            pr_auc_df = pd.DataFrame.from_dict(pr_auc_summary, orient='index', columns=['Mean PR AUC'])
            model_metrics_df.to_csv(subfolder_dir.joinpath(f'pr_auc_kfold_{target_variable}_{concatenated_condition}_{fn_suffix}.csv'))
            
            
            # Plotting metrics comparisons
            metrics_to_plot = ['Accuracy', 'Recall', 'F1 Score']
            for metric in metrics_to_plot:
                plt.figure(figsize=(10, 6))
                model_names = list(model_metrics[metric].keys())
                scores = list(model_metrics[metric].values())
                sns.barplot(x=scores, y=model_names)
                plt.xlabel('Mean ' + metric)
                plt.title(f'Cross-Validation {metric} Comparison')
                # Set x-axis limits to [0, 1]
                plt.xlim(0, 1)
                plt.tight_layout()
                plt.show()
                plt.savefig(subfolder_dir.joinpath(f'Model_Comparison_kfold_{target_variable}_{concatenated_condition}_{metric}_{fn_suffix}.png'))
                plt.close()

            
            
            # Plotting confusion matrices
            for name, matrix in conf_matrix_sum.items():
                plt.figure(figsize=(6, 4))
                sns.heatmap(matrix, annot=True, fmt='g')
                plt.title(f'Confusion Matrix: {name}')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.show()
                plt.savefig(subfolder_dir.joinpath(f'Confusion_Matrix_kfold_{target_variable}_{concatenated_condition}_{fn_suffix}_{name}.png'))
                plt.close()

            
            # Plot ROC curves
            plt.figure(figsize=(10, 6))
            for name in base_models.keys():
                # Aggregate and interpolate TPR for mean FPRs
                mean_fpr = np.linspace(0, 1, 100)
                mean_tpr = np.mean([np.interp(mean_fpr, fprs, tprs) 
                                    for fprs, tprs in zip(roc_data[name]['fpr'], roc_data[name]['tpr'])], axis=0)
                plt.plot(mean_fpr, mean_tpr, label=f'{name} (AUC = {roc_data[name]["auc"]:.2f})')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Mean ROC Curves')
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.show()
            plt.savefig(subfolder_dir.joinpath(f'ROC_curve_kfold_{target_variable}_{concatenated_condition}_{fn_suffix}.png'))
            plt.close()

            
            
            # Plot PR curves
            plt.figure(figsize=(10, 6))
            for name in base_models.keys():
                # Aggregate precision for mean recall
                mean_recall = np.linspace(0, 1, 100)
                mean_precision = np.mean([np.interp(mean_recall, recall[::-1], precision[::-1]) 
                                          for precision, recall in zip(pr_data[name]['precision'], pr_data[name]['recall'])], axis=0)
                plt.plot(mean_recall, mean_precision, label=f'{name} (AP = {pr_data[name]["pr_auc"]:.2f})')
                
            # Set x-axis and y-axis limits to [0, 1]
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Mean Precision-Recall Curves')
            plt.legend(loc='lower left')
            plt.tight_layout()
            plt.show()
            plt.savefig(subfolder_dir.joinpath(f'PR_curve_kfold_{target_variable}_{concatenated_condition}_{fn_suffix}.png'))
            
            plt.close('all')












# =============================================================================
# Stacking and cross validation not feasible because of low sample    
# =============================================================================

# Note: Feature importance cannot be directly retrieved from cross-validation. 
# It will need to be extracted separately, typically after fitting the model on the entire dataset.

## not doing stacking
# # Stacking classifier with cross-validation
# final_estimator = LogisticRegression(random_state=SEED)
# stacking_clf = StackingClassifier(
#     estimators=[(name, model) for name, model in base_models.items()],
#     final_estimator=final_estimator,
#     cv=skf
# )

# # Pipeline with scaling and stacking classifier
# stacking_pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('stacking', stacking_clf)
# ])

# # Evaluate the stacking classifier using bootstrapping in cross-validation
# stacking_cv_results = cross_validate(stacking_pipeline, X_reduced, y, cv=skf, scoring=('accuracy', 'f1_weighted'), return_train_score=True)

# print("---Stacking Classifier---")
# print(f"Mean Training Accuracy: {np.mean(stacking_cv_results['train_accuracy']):.4f}")
# print(f"Mean Validation Accuracy: {np.mean(stacking_cv_results['test_accuracy']):.4f}")
# print(f"Mean Training F1 Score: {np.mean(stacking_cv_results['train_f1_weighted']):.4f}")
# print(f"Mean Validation F1 Score: {np.mean(stacking_cv_results['test_f1_weighted']):.4f}")




## Train and evaluate each model using cross-validation and bootstrapping
# mean_fpr = np.linspace(0, 1, 100)
# tprs = {}
# aucs = {}

# for name, model in base_models.items():
#     plt.figure(figsize=(10, 8))

#     tprs[name] = []
#     aucs[name] = []
#     for i, (train_index, test_index) in enumerate(skf.split(X_reduced, y)):
#         # Bootstrap the training set
#         np.random.seed(i)  # Ensure reproducibility
#         boot_indices = np.random.choice(train_index, size=len(train_index), replace=True)
#         X_train_fold, X_test_fold = X_reduced.iloc[boot_indices], X_reduced.iloc[test_index]
#         y_train_fold, y_test_fold = y.iloc[boot_indices], y.iloc[test_index]

#         # Standardize features
#         scaler = StandardScaler()
#         X_train_fold = scaler.fit_transform(X_train_fold)
#         X_test_fold = scaler.transform(X_test_fold)
        
#         # Fit the model
#         model.fit(X_train_fold, y_train_fold)
        
#         # Predict probabilities
#         if hasattr(model, "predict_proba"):
#             probas_ = model.predict_proba(X_test_fold)[:, 1]
#         else:
#             probas_ = model.decision_function(X_test_fold)
#             probas_ = (probas_ - probas_.min()) / (probas_.max() - probas_.min())

#         # Compute ROC curve and area under the curve
#         fpr, tpr, thresholds = roc_curve(y_test_fold, probas_)
#         tprs[name].append(np.interp(mean_fpr, fpr, tpr))
#         tprs[name][-1][0] = 0.0
#         roc_auc = auc(fpr, tpr)
#         aucs[name].append(roc_auc)
#         plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {i+1} (AUC = {roc_auc:.2f})')

#     mean_tpr = np.mean(tprs[name], axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
#     std_auc = np.std(aucs[name])
    
#     plt.plot(mean_fpr, mean_tpr, color='b',
#               label=f'Mean ROC (AUC = {mean_auc:.2f} \u00B1 {std_auc:.2f})',
#               lw=2, alpha=.8)

#     plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
#     plt.xlim([-0.05, 1.05])
#     plt.ylim([-0.05, 1.05])
#     plt.xlabel('False Positive Rate')




