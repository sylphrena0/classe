#################################################
##### Superconductivity Regression Notebook #####
#################################################
# Trains models to predict critical temperatures based on features found with "*../code/get_featurizers.ipynb*". 
# Imports data from "*../data/supercon_feat.csv*", which is produced in *get_featurizers.ipynb*. 
# The orginal data is from the supercon database. This notebook is for testing single models.
# Compute-Farm version
# Author: Kirk Kleinsasser
#################################################

######################################################
#### Import Libraries / Define All Data Functions ####
######################################################
# %%
#general imports:
import argparse
import warnings #to suppress grid search warnings
import numpy as np 
import pandas as pd
import lolopy.learners #allows easy uncertainty
import matplotlib.pyplot as plt
import seaborn as sns #heatmaps
import forestci as fci #error for RFR - http://contrib.scikit-learn.org/forest-confidence-interval/index.html

#regression models:
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor, BayesianRidge
from sklearn.svm import SVR

#various ML tools:
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_predict, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, max_error

#imports custom libraries (shared functions)
import dependancies.shared_functions as sfn 
from dependancies.superlearner import get_superlearner as superlearner

###################################################
######## Define and Validate CLI Arguments ########
###################################################
# %% 
parser = argparse.ArgumentParser(description="A program that trains regression models for predicting superconductor critical temperatures.")
parser.add_argument('-fn', '--filename', action='store', default="supercon_features.csv", dest='filename', help='Select file to train models from /data/. Default is supercon_features.csv.')
parser.add_argument('-l', '--lr', action='store_true', dest='LR', help='Boolean option to enable the Linear Regression model.')
parser.add_argument('-s', '--svr', action='store_true', dest='SVR', help='Boolean option to enable the Support Vector Machines (Poly) model.')
parser.add_argument('-el', '--elastic', action='store_true', dest='ELASTIC', help='Boolean option to enable the Elastic Net Regression model.')
parser.add_argument('-dt', '--decisiontree', action='store_true', dest='DT', help='Boolean option to enable the Decision Tree Regression model.')
parser.add_argument('-rf', '--randomforest', action='store_true', dest='RFR', help='Boolean option to enable the Random Forest Regression model.')
parser.add_argument('-lrf', '--lolopyrandomforest', action='store_true', dest='LRFR', help='Boolean option to enable the lolopy Random Forest Regression model.')
parser.add_argument('-knn', '--knn', action='store_true', dest='KNN', help='Boolean option to enable the KNeighbors Regression model.')
parser.add_argument('-et', '--extratrees', action='store_true', dest='TREES', help='Boolean option to enable the Extra Trees Regression model.')
parser.add_argument('-sgd', '--stochastic', action='store_true', dest='SGD', help='Boolean option to enable the Stochastic Gradient Descent model.')
parser.add_argument('-by', '--bayes', action='store_true', dest='BAYES', help='Boolean option to enable the Bayesian Regression model.')
parser.add_argument('-sp', '--super', action='store_true', dest='SUPER', help='Boolean option to enable the Superlearner model.')

args = parser.parse_args()
#################################################
########### Setup Models for Training ###########
#################################################
# %% 
sfn.syncdir() #ensures working directory is inside code on compute farm
sfn.import_data(filename=args.filename,replace_inf=True) #import data without infinities

prefix = "Simulated " if "_sim" in args.filename else ""

models = ((args.LR, f"{prefix}Linear Regression", LinearRegression, {}),
            (args.SVR, f"{prefix}Support Vector Regression - Linear", SVR, {'kernel':'rbf', 'C':100, 'epsilon':0.1, 'gamma':0.1, 'degree':1}),
            (args.ELASTIC, f"{prefix}Elastic Net - Unoptimized", ElasticNet, {}),
            (args.ELASTIC, f"{prefix}Elastic Net - Optimized", ElasticNet, {'alpha':1e-05, 'l1_ratio':0.0}),
            (args.DT, f"{prefix}Decision Tree - Unoptimized", DecisionTreeRegressor, {}),
            (args.DT, f"{prefix}Decision Tree - Optimized", DecisionTreeRegressor, {'criterion':'poisson', 'max_depth':5, 'max_features':0.5}),
            (args.RFR, f"{prefix}Random Forest Regression", RandomForestRegressor, {}),
            (args.LRFR, f"{prefix}Random Forest Regression - Lolopy", lolopy.learners.RandomForestRegressor, {}),
            (args.KNN, f"{prefix}KNeighbors - Unoptimized", KNeighborsRegressor, {}),
            (args.KNN, f"{prefix}KNeighbors - Optimized", KNeighborsRegressor, {'metric':'manhattan', 'n_jobs':-1, 'n_neighbors':8}),
            (args.TREES, f"{prefix}Extra Trees - Unoptimized", ExtraTreesRegressor, {}),
            (args.TREES, f"{prefix}Extra Trees - Optimized", ExtraTreesRegressor, {'min_samples_leaf':1.0, 'min_samples_split':0.1, 'n_estimators':250, 'n_jobs':-1}),
            (args.SGD, f"{prefix}Stochastic Gradient Descent - Unoptimized", SGDRegressor, {}),
            (args.SGD, f"{prefix}Stochastic Gradient Descent - Optimized", SGDRegressor, {'alpha':1000.0, 'loss':'epsilon_insensitive', 'max_iter':1500, 'penalty':'l1'}),
            (args.BAYES, f"{prefix}Bayesian Regression - Unoptimized", BayesianRidge, {}),
            (args.BAYES, f"{prefix}Bayesian Regression - Optimized", BayesianRidge, {'alpha_init':1.2, 'lambda_init':0.0001}),
            (args.SUPER, f"{prefix}Superlearner", superlearner, {'X': sfn.train_data}))

# %%
######################################################
#################### Run Training ####################
######################################################

warnings.filterwarnings('ignore') #got tired of non-converging errors
for [enabled, model_name, regressor, parameters] in models: #optimize enabled models
    if enabled is True:
        print("Starting training on {}".format(model_name))
        sfn.evaluate_one(model_name, regressor, parameters, export=True)
    else:
        print(f"Skipping {model_name} as it not enabled.")
