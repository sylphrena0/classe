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
from dependancies.superlearner import get_superlearner as Superlearner

###################################################
######## Define and Validate CLI Arguments ########
###################################################
# %% 
parser = argparse.ArgumentParser(description="A program that trains regression models for predicting superconductor critical temperatures.")
parser.add_argument('-fn', '--filename', action='store', default="features.csv", dest='filename', help='Select file to train models from /data/. Default is supercon_features.csv.')
parser.add_argument('-o', '--optimized', action='store_true', default=False, dest='optimized', help='Boolean to enable/disable using optimized models. Defaults to True')
parser.add_argument('-u', '--uncert', action='store_true', default=True, dest='uncert', help='Enbable/disable uncertainty. Default is enabled.')
parser.add_argument('-um', '--uncertmethod', action='store', default="plus", dest='method', help='Select uncertainty method for mapie. Default is plus.')
parser.add_argument('-ci', '--forestci', action='store_true', default=False, dest='forestci', help='Boolean option to enable forestci uncertainty for random forests. Overrides --uncertmethod.')

args = parser.parse_args()
#################################################
########### Setup Models for Training ###########
#################################################
# %% 
sfn.syncdir() #ensures working directory is inside code on compute farm
sfn.import_data(filename=args.filename,replace_inf=True) #import data without infinities

prefix = "Simulated " if "_sim" in args.filename else ""

if args.optimized:
    optimized = "Optimized"
    
    #defines the optimized arguments for each model
    ELASTIC_ARGS = {'alpha':1e-05, 'l1_ratio':0.0}
    BAYES_ARGS = {}
    SVR_ARGS = {'C':1, 'epsilon':10, 'gamma':'auto', 'kernel':'linear'}
    RFR_ARGS = {'max_features': 10, 'n_estimators': 904}
    SP_ARGS = {'X': sfn.train_data}
    KNN_ARGS = {'metric':'manhattan', 'n_jobs':-1, 'n_neighbors':5}
    DT_ARGS = {'criterion':'poisson', 'max_features':0.5, 'random_state':43}
    TREES_ARGS = {'n_estimators': 708}

    #defines the models in a list of pairs of lists. The first item in a pair is the top graph in a column, the second is the bottom. The last item of a model is to enable uncert calc
    models  =  ((("Elastic Net Regression", ElasticNet, ELASTIC_ARGS, args.uncert),   ("Decision Tree Regression", DecisionTreeRegressor, DT_ARGS, args.uncert)),
                (("Bayesian Regression", BayesianRidge, BAYES_ARGS, args.uncert),     ("Random Forest Regression", RandomForestRegressor, RFR_ARGS, args.uncert)),
                (("Support Vector Machines", SVR, SVR_ARGS, args.uncert),             ("Extra Trees Regression", ExtraTreesRegressor, TREES_ARGS, args.uncert)),
                (("Linear Regression", LinearRegression, {}, args.uncert),                ("KNeighbors Regression", KNeighborsRegressor, KNN_ARGS, args.uncert)))
else:
    optimized = "Unoptimized"

    #defines the models in a list of pairs of lists. The first item in a pair is the top graph in a column, the second is the bottom. The last item of a model is to enable uncert calc
    models  =  ((("Elastic Net Regression", ElasticNet, {}, args.uncert),      ("Decision Tree Regression", DecisionTreeRegressor, {}, args.uncert)),
                (("Bayesian Regression", BayesianRidge, {}, args.uncert),      ("Random Forest Regression", RandomForestRegressor, {}, args.uncert)),
                (("Support Vector Machines", SVR, {}, args.uncert),            ("Extra Trees Regression", ExtraTreesRegressor, {}, args.uncert)),
                (("Linear Regression", LinearRegression, {}, args.uncert),   ("KNeighbors Regression", KNeighborsRegressor, {}, args.uncert)))

# %%
######################################################
#################### Run Training ####################
######################################################

warnings.filterwarnings('ignore') #got tired of non-converging errors
print("Starting training!")
sfn.evaluate(models, title=f'{prefix}Prediction vs. Actual Value (CV) - {optimized}', filename=f'{prefix}results_{optimized.lower()}', forestci=args.forestci, method=args.method)
# %%
