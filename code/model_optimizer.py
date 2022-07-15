################################################
##### Superconductivity Optimizer Notebook #####
################################################
# Trains models to predict critical temperatures based on features found with "*../code/get_featurizers.ipynb*". 
# Imports data from "*../data/supercon_feat.csv*", which is produced in *get_featurizers.ipynb*. The orginal data is from the supercon database. 
# Compute-Farm version
# Author: Kirk Kleinsasser
################################################

######################################################
### Import Libraries / Define Import Data Function ###
######################################################
# %% 
#general imports:
# import warnings #to suppress grid search warnings
import time
import argparse
import warnings
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns #heatmaps

#regression models:
# from mlens.ensemble import SuperLearner
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, SGDRegressor, BayesianRidge
from sklearn.svm import SVR

#various ML tools:
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, r2_score, mean_absolute_error, mean_squared_error
# from skopt import BayesSearchCV #bayesian optimization

#imports custom libraries (shared functions)
import dependancies.shared_functions as sfn

###################################################
######## Define and Validate CLI Arguments ########
###################################################
# %% 
sfn.syncdir() #ensures working directory is inside code on compute farm

parser = argparse.ArgumentParser(description="A program that optimizes regression models for predicting superconductor critical temperatures.")
parser.add_argument('-s', '--samplesize', action='store', dest='limit', default=1000, help='Limit the GridSearch Data Sample Size. Value must be \'all\' or a number between 0 and 16414')
parser.add_argument('-sv', '--svr', action='store_true', dest='SVR', help='Boolean option to enable the Support Vector Machines (Linear) model.')
parser.add_argument('-svp', '--svrpoly', action='store_true', dest='SVR_POLY', help='Boolean option to enable the Support Vector Machines (Poly) model.')
parser.add_argument('-el', '--elastic', action='store_true', dest='ELASTIC', help='Boolean option to enable the Elastic Net Regression model.')
parser.add_argument('-dt', '--decisiontree', action='store_true', dest='DT', help='Boolean option to enable the Decision Tree Regression model.')
parser.add_argument('-rf', '--randomforest', action='store_true', dest='RFR', help='Boolean option to enable the Random Forest Regression model.')
parser.add_argument('-knn', '--knn', action='store_true', dest='KNN', help='Boolean option to enable the KNeighbors Regression model.')
parser.add_argument('-et', '--extratrees', action='store_true', dest='TREES', help='Boolean option to enable the Extra Trees Regression model.')
parser.add_argument('-sgd', '--stochastic', action='store_true', dest='SGD', help='Boolean option to enable the Stochastic Gradient Descent model.')
parser.add_argument('-by', '--bayes', action='store_true', dest='BAYES', help='Boolean option to enable the Bayesian Regression model.')

args = parser.parse_args()

if 0 < int(args.limit) < 16414:
    pass
elif str(args.limit) == 'all':
    args.limit = 16414
else:
      raise Exception("Invalid GridSearch Data Sample Size Limit. Value must be 'all' or a number between 0 and 16414.") #i am once again asking for a valid input :(

#####################################################
########### Setup Models for GridSearchCV ###########
#####################################################
# %% 

snf.import_data(replace_inf=False) #grab data

#drop data that will not be used for optimization after shuffle, to limit defined in function
limit = int(limit)
train_data = snf.train_data.iloc[:limit]
test_data = snf.test_data.iloc[:limit]
train_target = snf.train_target.iloc[:limit]
test_target = snf.test_target.iloc[:limit]

#get number of rows and columns for use in parameters
n_features = snf.data.shape[1]
n_samples = snf.data.shape[0]

#define parameters that will be searched with GridSearchCV
SVR_PARAMETERS = {"kernel": ["poly","rbf","sigmoid"], "degree": np.arange(1,10,2), "C": np.linspace(0,1000,5), "epsilon": np.logspace(-3, 3, 5),
                    "gamma": [1.00000000e-03, 5.99484250e-02, 4.64158883e-01, 3.59381366e+00, 1.00000000e+01, "scale", "auto"]}
SVR_POLY_PARAMETERS = {"C": np.linspace(0,1000,5), "epsilon": np.logspace(-3, 3, 5), 
                    "gamma": [1.00000000e-03, 5.99484250e-02, 4.64158883e-01, 3.59381366e+00, 1.00000000e+01, "scale", "auto"]}
ELASTIC_PARAMETERS = {"alpha": np.logspace(-10, 2, 5), 'l1_ratio': np.arange(0, 1, 0.1)}
DT_PARAMETERS = {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], 'max_depth': [None, 1, 3, 5, 7], 
                    'max_features': [None, 'sqrt', 'log2', 0.3, 0.5, 0.7, n_features//2, n_features//3, ],
                    'min_samples_split': [3, 2, 0.3, 0.5, n_samples//2, n_samples//3, n_samples//5], 
                    'min_samples_leaf':[1, 0.3, 0.5, n_samples//2, n_samples//3, n_samples//5]}
RFR_PARAMETERS = {'max_features': ['auto', 1, 2, 3, 4, 5], 'n_estimators': np.linspace(1,1000,20,dtype=int)}
KNN_PARAMETERS = {'n_neighbors': np.linspace(1,15,5,dtype=int), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 
                    'metric':['euclidean', 'manhattan']}
TREES_PARAMETERS = {'n_estimators': np.linspace(1,750,15,dtype=int)} 
SGD_PARAMETERS = {'loss': ['hinge', 'log_loss', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                    'penalty': ['l1', 'l2', 'elasticnet'], "alpha": np.logspace(-4, 5, 5)}
BAYES_PARAMETERS = {'alpha_init':[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.9], 'lambda_init': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-9]}

models =   [[args.SVR, "Support Vector Machines (Linear)", SVR, SVR_PARAMETERS, {'max_iter': -1}],
            [args.SVR_POLY, "Support Vector Machines (Poly)", SVR, SVR_POLY_PARAMETERS, {'max_iter': -1}],
            [args.ELASTIC, "Elastic Net Regression", ElasticNet, ELASTIC_PARAMETERS, {'fit_intercept': True}],
            [args.DT, "Decision Tree Regression", DecisionTreeRegressor, DT_PARAMETERS, {'random_state': 43}],
            [args.RFR, "Random Forest Regression", RandomForestRegressor, RFR_PARAMETERS, {'bootstrap': True, 'n_jobs': -1}],
            [args.KNN, "KNeighbors Regression", KNeighborsRegressor, KNN_PARAMETERS, {'n_jobs': -1}],
            [args.TREES, "Extra Trees Regression", ExtraTreesRegressor, TREES_PARAMETERS, {'n_jobs': -1}],
            [args.SGD, "Stochastic Gradient Descent", SGDRegressor, SGD_PARAMETERS, {'fit_intercept': True, 'max_iter': 1500}],
            [args.BAYES, "Bayesian Regression", BayesianRidge, BAYES_PARAMETERS, {'fit_intercept': True}]]

def optimize_model(model_name, regressor, parameters, fixed_params): #performs grid search on a given model with specified search and fixed model parameters and saves results to csv
    global results #variables that we want to define globally (outside of this funtion)
    #this function will allow us to use multiprocessing to do multiple grid searches at once.
    try: #try-excepts handles errors without ending process and allows us to read the error later on
        start_time = time.time() #sets start time for function so we can record processing time
        #define model, do grid search
        search = GridSearchCV(regressor(**fixed_params), #model
                        param_grid = parameters, #hyperparameters
                        scoring='r2', #metrics for scoring
                        return_train_score = False, #we want test score
                        cv = 3, #number of folds
                        n_jobs = -1, #amount of threads to use
                        # refit = 'r2', #metric we are optimizing (no need to set for single metric scorring)
                        verbose = 1) #how much output to send while running

        search.fit(train_data, train_target) #fit the models
        return (model_name, search.best_estimator_, search.best_params_, "Best Score: " + str(search.best_score_), "Time Elapsed: " + str(time.time() - start_time)) #record results
    except Exception as error: #catch any issues and record them
        return (model_name, "ERROR", "ERROR", error) #record errors

####################################################
#################### Run Search ####################
####################################################
# %% 

results = []
warnings.filterwarnings('ignore') #got tired of non-converging errors
for [enabled, model_name, regressor, parameters, fixed_params] in models: #optimize enabled models
    if enabled is True:
        print("Starting GridSearchCV on {}".format(model_name))
        results.append(optimize_model(model_name, regressor, parameters, fixed_params))
    else:
        print(f"Skipping {model_name} as it is not enabled.")

result_df = pd.DataFrame(results)
result_df.to_csv('../data/optimize.results.csv') #saves data to './optimize_results.csv'
