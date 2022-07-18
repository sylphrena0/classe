################################################
############ Superlearner Functions ############
################################################
# This python file creates functions needed to run a superlearner and allows us to run a mlens superlearner with the same syntax
# as sklearn models, using get_superlearner(). This file is used for ../training_bulk.ipynb and ../training_single.ipynb.
#
# Author: Kirk Kleinsasser
################################################

################################################
############ Define Import Function ############
################################################
#regression models:
from mlens.ensemble import SuperLearner
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, SGDRegressor, BayesianRidge
from sklearn.svm import SVR

#sklearn scoring metrics:
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, max_error

##############################################
############## Define Functions ##############
##############################################
# create a list of base-models
def get_models():
	models = list()
	models.append(SVR())
	models.append(ElasticNet(alpha=1e-05, l1_ratio=0.0))
	models.append(DecisionTreeRegressor())
	models.append(RandomForestRegressor())
	models.append(KNeighborsRegressor(metric='manhattan', n_jobs=-1, n_neighbors=8))
	# models.append(ExtraTreesRegressor())
	# models.append(SGDRegressor(alpha=1000.0, loss='epsilon_insensitive', max_iter=1500, penalty='l1'))
	# models.append(BayesianRidge(alpha_init=1.2, lambda_init=0.0001))

	return models

def get_superlearner(X, scorer=r2_score, folds=10, shuffle=True):
	ensemble = SuperLearner(scorer=scorer, folds=folds, shuffle=shuffle, sample_size=len(X), n_jobs=-1)
	ensemble.add(get_models()) #add base models
	ensemble.add_meta(LinearRegression()) #add meta model

	return ensemble