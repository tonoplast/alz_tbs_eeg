# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 19:33:20 2023

@author: SChung
"""


from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
import scipy.stats as stats
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.feature_selection import VarianceThreshold

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from src.utils.modelling_tools import run_pca_to_remove_correlated_features, select_features, compute_vif, plot_actual_vs_predicted_value, plot_feature_importance
from src.utils.modelling_tools import logged_box_plot, get_corr, eval_regression_models


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


SEED = 9
run_pca=False
plot_actual_vs_predicted=True
plot_feature_importance_bool=True
RF_HYPERPARAM_TUNING=False
percentile_cutoff = 100


df_roi_bl_long = turn_df_wide_to_long(df_roi_bl)
df_roi_delta_long = turn_df_wide_to_long(df_roi_delta)

## merging all the data together
# df = df_main.merge(df_roi_bl_long, on=['ID','COND'], how='inner').merge(df_roi_delta_long, on=['ID','COND'], how='inner')
df = df_main.merge(df_roi_bl_long, on=['ID','COND'], how='inner')



x_col = list(df.drop(columns=['ID','COND','change_list_recall', 'Cogstate_list_recall_W6', 'mean_gamma_connectivity_END_1']))
y_col = ['change_list_recall']


# =============================================================================
# Check for (multi)collinearity
# =============================================================================

## heatmap
plt.figure(figsize=(14, 8))
mask = np.triu(np.ones_like(df[x_col].corr(), dtype=bool))
relation = df[x_col].corr()
relation_index = relation.index
sns.heatmap(df[relation_index].corr(), annot=True, mask=mask) #, vmin=-1, vmax=1)
plt.title('Cross-correlation')
plt.tight_layout()

# check_vif = compute_vif(df, x_col).sort_values('VIF', ascending=False)

X = df[x_col]
y = df[y_col]

from sklearn.feature_selection import RFE
model = LinearRegression()
rfe = RFE(model, n_features_to_select=5) # choose 5 best features
fit = rfe.fit(X, y)
selected_features = rfe.support_
feature_ranking = rfe.ranking_

selected_feature_indices = [i for i, selected in enumerate(selected_features) if selected]
selected_features_names = [X.columns[i] for i in selected_feature_indices]

print("Selected Features:", selected_features_names)


# =============================================================================
# split
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=SEED)
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, mutual_info_regression)



# Create a list of feature names and their corresponding scores
feature_scores = [(X.columns[i], fs.scores_[i]) for i in range(len(fs.scores_))]

# Sort the list by scores in descending order
sorted_feature_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)

# Print and plot the sorted feature scores
# for feature, score in sorted_feature_scores:
#     print('Feature - %s: %f' % (feature, score))

# Plot the sorted scores
sorted_features, sorted_scores = zip(*sorted_feature_scores)
plt.figure(figsize=(7, 6))
plt.bar(sorted_features, sorted_scores)
plt.xticks(rotation=90)
plt.title('Feature selection scores (sorted)')
plt.tight_layout()
plt.show()



# Calculate the number of features to keep (10% of total features)
num_features_to_keep = int(len(sorted_feature_scores) * (percentile_cutoff / 100))

# Get the top X % features
top_X_percent_features = sorted_feature_scores[:num_features_to_keep]


# Plot the top 10% features and their scores
top_X_percent_features, top_X_percent_scores = zip(*top_X_percent_features)

# plt.figure(figsize=(7, 6))
# plt.bar(top_X_percent_features, top_X_percent_scores)
# plt.xticks(rotation=90)
# plt.title(f'Top {percentile_cutoff}% of Feature selection scores')
# plt.tight_layout()
# plt.show()
                
X = X[list(top_X_percent_features)]


if run_pca:

    # =============================================================================
    # ## PCA    
    # =============================================================================
    
    
    df_pca, \
        df_top_features_and_exp_variance, \
            loadings_df = run_pca_to_remove_correlated_features(X, 
                                                                excluded_columns=[], 
                                                                variance_threshold=0.95, 
                                                                top_n_component_per_pc=10, 
                                                                show_plot=False) 
    
    ## saving so that we know what PCA consists of

    pca_comps = [i for i in df_pca.columns if 'PC' in i]

                
    # =============================================================================
    # Simple linear regression (visualisation purposes)
    # =============================================================================
    ## can be useful for understanding the shape of the data
    ## usually with many features, all features are unlikely to have linear relationship
    
    # get_corr(df, dv, ivs)
    
    # =============================================================================
    # Check for (multi)collinearity
    # =============================================================================

    # seems acceptable..
    considered_features = pca_comps.copy()
    check_vif = compute_vif(df_pca, considered_features).sort_values('VIF', ascending=False)
    features_to_keep = list(df_pca.columns)
    

    # =============================================================================
    # Split data - training / test
    # =============================================================================
    
    X = df_pca[features_to_keep]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=SEED)


train_index = X_train.index.values
test_index = X_test.index.values
            
# =============================================================================
# Scaling - mainly because we are comparing different models below
# but also, the range are quite different. I think for RF, it may not be
# necessary, but I like scaling. Will leave this up to you.
# =============================================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# =============================================================================
# Model selection
# =============================================================================

# polynomial is a bit special and need to pipe it through first
# degree=2
# polyreg = make_pipeline(PolynomialFeatures(degree),LinearRegression())
polyreg_degree_2 = make_pipeline(PolynomialFeatures(2),LinearRegression())
polyreg_degree_3 = make_pipeline(PolynomialFeatures(3),LinearRegression())
polyreg_degree_4 = make_pipeline(PolynomialFeatures(4),LinearRegression())

regressors = {
    "XGBRegressor":  XGBRegressor(
                                    learning_rate=0.1,          # Adjust as needed
                                    n_estimators=100,          # Adjust as needed
                                    max_depth=5,               # Adjust as needed
                                    min_child_weight=1,        # Adjust as needed
                                    subsample=0.8,             # Adjust as needed
                                    colsample_bytree=0.8,      # Adjust as needed
                                    reg_alpha=0.1,             # Adjust as needed
                                    reg_lambda=0.1,            # Adjust as needed
                                    random_state=SEED            # Set a random seed for reproducibility
                                ),
    "RandomForestRegressor": RandomForestRegressor(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "LinearRegression": LinearRegression(),
    # "RANSAC": RANSACRegressor(),
    "LASSO": Lasso(alpha=0.1),
    # "Polynomial": polyreg,
    
    # "GaussianProcessRegressor": GaussianProcesssRegressor(), ## somehow it keeps crashing
    "SVR": SVR(),
    "NuSVR": NuSVR(),
    "LinearSVR": LinearSVR(),
    "KernelRidge": KernelRidge(),
    "Ridge":Ridge(),
    "TheilSenRegressor": TheilSenRegressor(),
    "HuberRegressor": HuberRegressor(),
    "PassiveAggressiveRegressor": PassiveAggressiveRegressor(),
    "ARDRegression": ARDRegression(),
    "BayesianRidge": BayesianRidge(),
    "ElasticNet": ElasticNet(),

    "Polynomial_2": polyreg_degree_2,
    "Polynomial_3": polyreg_degree_3,
    "Polynomial_4": polyreg_degree_4,
}


# =============================================================================
# loop and ranking based on rmse_cv (10-fold cross-validation)
# =============================================================================

df_models = pd.DataFrame(columns=['model', 'run_time', 'r-squared', 'mae', 'rmse', 'rmse_cv', 'pearsonr2'])


for key in regressors:

    print('*',key)

    start_time = time.time()
    
    regressor = regressors[key]
    model = regressor.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # plot check
    if plot_actual_vs_predicted:
        plot_actual_vs_predicted_value(test_index, y_pred, y_test, f'Actual vs Predicted: {key}')    
    
    if key in ['XGBRegressor','RandomForestRegressor','DecisionTreeRegressor']:
        if plot_feature_importance_bool:

            plot_feature_importance(X, regressor, key)
            # importance_model = pd.Series(regressor.feature_importances_, index=X.columns).sort_values()
    
    scores = cross_val_score(model, 
                             X_train, 
                             y_train,
                             scoring="neg_mean_squared_error", 
                             cv=10)
    
    _r2_score, _mae, _mse, _rmse, _pearsonr2 = eval_regression_models(model, y_test, y_pred)
            
    row = {'model': key,
           'run_time': format(round((time.time() - start_time)/60, 3)),
           'r-squared': round(_r2_score, 3),
           'mae': round(_mae, 3),
           'mse': round(_mse, 3),
           'rmse': round(_rmse, 3),
           'rmse_cv': round(np.mean(np.sqrt(-scores)), 3),
           'pearsonr2': round(_pearsonr2[0], 3)
    }
    
    df_dict = pd.DataFrame([row])
    df_models = pd.concat([df_models, df_dict], ignore_index=True)


# see scoring
print(df_models.sort_values(by='rmse_cv', ascending=True))


# =============================================================================
# plot (top performing one)
# =============================================================================
# regressor = XGBRegressor(random_state=SEED)
# model = regressor.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# eval_models(model, y_test, y_pred, True)
# plot_actual_vs_predicted_value(test_index, y_pred, y_test, 'Actual vs Predicted: XGBoostRegressor')    
# plot_feature_importance(regressor, 'XGBoost')


if RF_HYPERPARAM_TUNING:
    # =============================================================================
    # Hyperparameter tuning -- THIS TAKES VERY LONG
    # Try out whatever you think is best
    # =============================================================================
    model.get_params()
    
    def eval_best_rf_model(best_model, X_test, y_test):
        y_pred = best_model.predict(X_test)
        eval_regression_models(best_model, y_test, y_pred, True)
        plot_actual_vs_predicted_value(test_index, y_pred, y_test, 'Actual vs Predicted: RandomForestRegressor')    
        plot_feature_importance(best_model.best_estimator_, 'RandomForest (Hyperparameter-tuned)')
        
       
        predicted = best_model.best_estimator_.predict(X_train)
        residuals = y_train-predicted
        
        fig, ax = plt.subplots()
        ax.scatter(y_train, residuals)
        ax.axhline(lw=2,color='black')
        ax.set_xlabel('Observed')
        ax.set_ylabel('Residual')
        plt.show()
        
        df_gs = pd.DataFrame(data=best_model.cv_results_)
        df_gs_plot = df_gs[['mean_test_score',
                            'param_max_leaf_nodes',
                            'param_max_depth']].sort_values(by=['param_max_depth',
                                                                'param_max_leaf_nodes'])
        
        
        
        fig,ax = plt.subplots()
        sns.pointplot(data=df_gs_plot,
                      y='mean_test_score',
                      x='param_max_depth',
                      hue='param_max_leaf_nodes',
                      ax=ax)
        ax.set(title="Effect of Depth and Leaf Nodes on Model Performance")
    
    
    # =============================================================================
    #     # random search -- quick
    # =============================================================================
    random_grid  = dict(
                    bootstrap = [True, False],
                    criterion = ['squared_error'],
                    max_depth = [None] + [int(x) for x in np.linspace(10, 100, num = 10)],
                    min_samples_leaf = [1, 2, 3, 4],
                    min_samples_split = [2, 4, 6, 8, 10],
                    max_leaf_nodes = [None, 2, 4, 8, 10, 12],
                    max_features = [None, 'sqrt', 'log2'], 
                    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 500, num = 50)],
                    )  
    
    rf = RandomForestRegressor(random_state=SEED)
    rf_random = RandomizedSearchCV(estimator = rf, 
                                   param_distributions = random_grid, 
                                   n_iter = 200, 
                                   cv = 3, 
                                   verbose = 2, 
                                   random_state = SEED, 
                                   n_jobs = -1)
    
    best_model_rs = rf_random.fit(X_train, y_train)   
    print('Optimum parameters:')
    pprint(best_model_rs.best_params_)
    
    eval_best_rf_model(best_model_rs, X_test, y_test)


    # # =============================================================================
    # #     # grid search -- takes long
    # # =============================================================================
    # param_grid = dict(
    #                 criterion=['squared_error'],
    #                 max_depth=[None, 3, 5, 7, 9], 
    #                 min_samples_leaf=np.linspace(0.1, 0.5, 5, endpoint=True),
    #                 min_samples_split=[0.01, 0.03, 0.05, 0.1, 0.2],
    #                 max_leaf_nodes=[None, 2, 4, 8, 10, 12],
    #                 # max_features=list(range(1, X.shape[1])), 
    #                 max_features=[None, 'sqrt', 'log2'], 
    #                 # n_estimators=[1, 2, 4, 6, 16, 32, 64, 100, 120, 140, 160],
    #                 n_estimators=[20, 50, 75, 100, 200],
    
    #                 )

    # model = RandomForestRegressor(random_state=SEED)
    
    # grid_search = GridSearchCV(estimator = model,
    #                            param_grid = param_grid,
    #                            scoring = 'neg_root_mean_squared_error',
    #                            n_jobs = -1,
    #                            verbose = 2,
    #                            cv = 3
    #                            )
    
    
    # best_model_gs = grid_search.fit(X_train, y_train)
    # print('Optimum parameters:', best_model_gs.best_params_)
    
 
    # # Optimum parameters: {'criterion': 'squared_error', 
    # #                      'max_features': 'sqrt', 
    # #                      'max_leaf_nodes': 8, 
    # #                      'min_samples_leaf': 0.1, 
    # #                      'min_samples_split': 0.01, 
    # #                      'n_estimators': 50}
    
    # eval_best_rf_model(best_model_gs, X_test, y_test)
