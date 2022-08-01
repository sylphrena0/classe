#################################################
##### Superconductivity Regression Notebook #####
#################################################
# Trains models to predict critical temperatures based on features found with "*../code/get_featurizers.ipynb*". 
# Imports data from "*../data/features.csv*", which is produced in *get_featurizers.ipynb*. 
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
parser.add_argument('-fn', '--filename', action='store', default="features.csv", dest='filename', help='Select file to train models from /data/. Default is supercon_features.csv.')
parser.add_argument('-fi', '--featureimportance', action='store_true', dest='fi', help='Boolean option to enable exporting feature importance.')
parser.add_argument('-a', '--all', action='store_true', dest='all', help='Boolean option to enable all regression models. Overrides individual toggles. Does not include lolopy model.')
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

suffix = " (No Outliers)" if "_outliers" in args.filename else ""

models = (((args.LR, args.all), "Linear Regression", LinearRegression, {}),
            ((args.SVR, args.all), "Support Vector Regression - Linear", SVR, {'kernel':'rbf', 'C':100, 'epsilon':0.1, 'gamma':0.1, 'degree':1}),
            ((args.ELASTIC, args.all), "Elastic Net - Unoptimized", ElasticNet, {}),
            ((args.ELASTIC, args.all), "Elastic Net - Optimized", ElasticNet, {'alpha':1e-05, 'l1_ratio':0.0}),
            ((args.DT, args.all), "Decision Tree - Unoptimized", DecisionTreeRegressor, {}),
            ((args.DT, args.all), "Decision Tree - Optimized", DecisionTreeRegressor, {'criterion':'poisson', 'max_depth':5, 'max_features':0.5}),
            ((args.RFR, args.all), "Random Forest Regression", RandomForestRegressor, {}),
            ((args.LRFR,), "Random Forest - Lolopy", lolopy.learners.RandomForestRegressor, {}), #note that the all argument does not enable lolopy 
            ((args.KNN, args.all), "KNeighbors - Unoptimized", KNeighborsRegressor, {}),
            ((args.KNN, args.all), "KNeighbors - Optimized", KNeighborsRegressor, {'metric':'manhattan', 'n_jobs':-1, 'n_neighbors':8}),
            ((args.TREES, args.all), "Extra Trees - Unoptimized", ExtraTreesRegressor, {}),
            ((args.TREES, args.all), "Extra Trees - Optimized", ExtraTreesRegressor, {'min_samples_leaf':1.0, 'min_samples_split':0.1, 'n_estimators':250, 'n_jobs':-1}),
            ((args.SGD, args.all), "Stochastic Gradient Descent - Unoptimized", SGDRegressor, {}),
            ((args.SGD, args.all), "Stochastic Gradient Descent - Optimized", SGDRegressor, {'alpha':1000.0, 'loss':'epsilon_insensitive', 'max_iter':1500, 'penalty':'l1'}),
            ((args.BAYES, args.all), "Bayesian Regression - Unoptimized", BayesianRidge, {}),
            ((args.BAYES, args.all), "Bayesian Regression - Optimized", BayesianRidge, {'alpha_init':1.2, 'lambda_init':0.0001}),
            ((args.SUPER, args.all), "Superlearner", superlearner, {'X': sfn.train_data}))

# %%
######################################################
#################### Run Training ####################
######################################################

warnings.filterwarnings('ignore') #got tired of non-converging errors
for [enabled, model_name, regressor, parameters] in models: #optimize enabled models
    model_name += suffix
    if True in enabled: #if model is enabled or all models are enabled
        print("Starting training on {}".format(model_name))
        sfn.evaluate_one(model_name, regressor, parameters, image=True, export_feat_importance=args.fi)
    else:
        print(f"Skipping {model_name} as it is not enabled.")
