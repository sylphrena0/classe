################################################
########### General Shared Functions ###########
################################################
# This python file creates functions needed to import data and evaluate models. This limits redundancy 
# and this file can be imported from our code in .. with import dependancies.shared_functions as sfn. 
# This file is used for ../training_bulk.ipynb and ../training_single.ipynb.
#
# Author: Kirk Kleinsasser
################################################

###############################################
########## Import Required Libraries ##########
###############################################
import warnings #to suppress grid search warnings
import os
import re #regex
import numpy as np 
import pandas as pd
import lolopy.learners #allows easy uncertainty
import matplotlib.pyplot as plt
import seaborn as sns #heatmaps
import forestci as fci #error for RFR - http://contrib.scikit-learn.org/forest-confidence-interval/index.html

#various ML tools:
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, max_error
from sklearn.ensemble import RandomForestRegressor
from mapie.regression import MapieRegressor
from mapie.metrics import regression_mean_width_score as regression_mws

################################################
############# Define Sync Function #############
################################################
def syncdir():
    if os.getcwd().endswith('kvk23/queue'): #if working directory is queue
        os.chdir("code") #enter into code (prevents conflicts with running shared function locally)

################################################
############ Define Import Function ############
################################################
def import_data(filename="supercon_features.csv", replace_inf=False, drop=None, split=True):
    '''
    Imports the data from get_featurizers. Drop argument can be a list of columns to drop or a string like
    '''
    global data, target, train_data, test_data, train_target, test_target #variables that we want to define globally (outside of this funtion)
    data = pd.DataFrame(pd.read_csv(f'../data/{filename}')) #loads data produced in get_featurizer.ipynb

    if replace_inf: #replaces values of infinity with NaN if replace_inf is True
        data.replace([np.inf, -np.inf], np.nan, inplace=True) 

    #TODO: debug feaurizers - NaN is entered when there is an error in the featurizer
    data.drop(['name','Unnamed: 0', 'composition'], axis=1, inplace=True) #drop columns irrelevant to training
    if drop is not None:
        data.drop(drop, axis=1, inplace=True) #drop columns, if specified
    data = data[data.columns[data.notnull().any()]] #drop columns that are entirely NaN (12 columns) 

    for col in data: #replaces NaN with zeros
        data[col] = pd.to_numeric(data[col], errors ='coerce').fillna(0).astype('float')

    if split: #this is for our feature anaylsis notebook. to drop data based on Tc without a mess of operator if-elif statements, we need to not pop Tc or split our data yet
        target = data.pop('Tc') #remove target (critical temp) from data
        train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.15, random_state=43, shuffle=True) #creates a test train split, with shuffle and random state for reproducibility 

###############################################
######### Define Evaluation Functions #########
###############################################
def evaluate_one(model_name, model, parameters, uncertainty=True, method="plus", forestci=False, export_feat_importance=False, image=False, csv=False, show=True, maxexpected=135):
    """
    Defines function that trains a model to predict critical temp and plots with metrics and optional uncertainty.
    Uncertainty and forestci arguments override method specifications. forestci is much faster than mapie and is only applicable to random forest models.
    Maxexpected argument is to change the length of the expected value dotted line. Can also export feature importance, unlike evaluate().
    """
    global train_data, train_data, test_data, test_target #we need these variables and don't want to pass them as arguments
    warnings.filterwarnings("ignore", category=FutureWarning)
    with plt.rc_context({'xtick.color':'white', 'ytick.color':'white','axes.titlecolor':'white','figure.facecolor':'#1e1e1e','text.color':'white','legend.labelcolor':'black'}):
        plt.title(f"{model_name} - Prediction vs. Actual Value (CV)", color='white')
        regressor = model(**parameters)
        save_name = re.sub(" - | ", "_", re.sub("\(|\)", "", model_name)).lower() #first removes paranthesis, then replaces " - " or " " with underscores to make a nice savename
        results = pd.DataFrame(columns=("model","mean_squared_error","mean_absolute_error","max_error","r2_score","mapie_eff_mean_width"))

        if uncertainty and not forestci and method != "prefit": #uncertainty calculations need magie training if not forestci/prefit mapie
            mapie_regressor = MapieRegressor(estimator=regressor, method=method) #unpacks model and params
            if model_name.startswith(("Superlearner", "Random Forest - Lolopy")): #need to get values for these models
                mapie_regressor.fit(train_data.values, train_target.values) #fit the model
            else:
                mapie_regressor.fit(train_data, train_target) #fit the model
            model_pred, model_pis = mapie_regressor.predict(test_data, alpha=0.05) #make predictions on test data
        else: #no need for uncertainty calculations during training, use sklearn
            if model_name in ("Superlearner", "Random Forest - Lolopy"): #need to get values for these models
                regressor.fit(train_data.values, train_target.values) #fit the model
            else:
                regressor.fit(train_data, train_target) #fit the model
            model_pred = regressor.predict(test_data) #make predictions on test data
        
        if export_feat_importance and hasattr(regressor, 'feature_importances_'): #if we are exporting feature importance and model has attribute "feature_importances_"
            feat_columns = data.columns.tolist()
            feat_importance = pd.DataFrame(feat_columns, columns=('Feature',)) #get feature importances from model
            feat_importance['Importance'] = regressor.feature_importances_.astype(float)
            feat_importance = feat_importance.sort_values('Importance', ascending=False)
            feat_importance.to_csv(f'../data/importance/{save_name}_importance.csv', index=False) #save csv
        elif export_feat_importance:
            warnings.warn("Cannot calculate feature importance, this model might not have feature_importances_, or you may be trying to run with uncertainty calculations, which is not supported.", category=Warning)
            
    
        mse = round(mean_squared_error(test_target, model_pred),3) #find mean square error
        mxe = round(max_error(test_target, model_pred),3)
        mae = round(mean_absolute_error(test_target, model_pred),3) #find mean square error
        r2 = round(r2_score(test_target, model_pred),3) #find r2 score

        #make our plot - with plt.rc_context sets theme to look good in dark mode
        difference = np.abs(test_target - model_pred) #function that finds the absolute difference between predicted and actual value
        im = plt.scatter(test_target, model_pred, cmap='plasma_r', norm=plt.Normalize(0, 120), c=difference, label="Critical Temperature (K)", zorder=2) #create scatter plot of data 
        plt.plot((0,maxexpected), (0,maxexpected), 'k--', alpha=0.75, zorder=3) #add expected line. Values must be changed with different data to look good
        if uncertainty: #plot error bars
            if forestci and model is not RandomForestRegressor:
                raise NameError("RandomForestRegressor must be selected to use forestci")
            elif forestci:
                model_unbiased = fci.random_forest_error(regressor, train_data, test_data, calibrate=False) #NOTE: forestci calibration is disabled as there is a bug in the code (they use a too small datatype)
                yerror = np.sqrt(model_unbiased)
                print(yerror.shape)
            elif method == "prefit":
                raise NameError("Prefit method is not implemented in this program as our current implementation have error bars that do not align with the test data")
                # mapie = MapieRegressor(estimator=model, cv="prefit").fit(cal_data, cal_target) #important: calibration data must be different from training data!
                # pred_interval = pd.DataFrame(mapie.predict(test_data, alpha=.05)[1].reshape(-1,2), index=test_data.index, columns=["lower", "upper"]) #get interval predictions on test data, with alpha=5%
                # yerror = pred_interval.values.reshape(2,-1)
            else:
                #model_pis contains absolute points for upper/lower bounds. We need absolute error, like (3, 3) for ± 3:
                yerror = np.abs(model_pis[:,:,0].transpose() - np.tile(model_pred, (2, 1))) #error must be in shape (n, 2) for errorbars
                mws = round(regression_mws(model_pis[:,:,0][:,0],model_pis[:,:,0][:,1]),3) #generate mean width score metric from mapie data
            plt.errorbar(test_target, model_pred, yerr=yerror, fmt="none", ecolor="black", alpha=0.5, zorder=1, label="Prediction Intervals")

        results.loc[len(results.index)] = (model_name,mse,mae,mxe,r2,mws)
        
        plt.title(model_name, c='white')
        plt.xlabel('Actual Value', c='white')
        plt.ylabel('Prediction', c='white')
        plt.annotate(f'R2: {r2}', xy = (0, -0.15), xycoords='axes fraction', ha='left', va="center", fontsize=10) #add footnote with R2 
        plt.annotate(f'MWS: {mws}' if mws else f'MXE: {mxe}', xy = (0, -0.20), xycoords='axes fraction', ha='left', va="center", fontsize=10) #add footnote with R2 
        plt.annotate(f'MAE: {mae}', xy = (1.0, -0.20), xycoords='axes fraction', ha='right', va="center", fontsize=10) #add footnote with MAE
        plt.annotate(f'MSE: {mse}', xy = (1.0, -0.15), xycoords='axes fraction', ha='right', va="center", fontsize=10) #add footnote with MSE
        plt.legend()
        plt.colorbar().set_label(label="Difference from Actual (K)", color='white') #using .set_label() as colorbar() does accept color arguments
        if image:
            plt.savefig(f'../data/indvidual_results/{save_name}.png', bbox_inches='tight')
        if csv:
            results.to_csv(f'../data/indvidual_results/{save_name}.csv', index=False)
        if show:
            plt.show()
        plt.clf()

def evaluate(models, title, filename='results', method="plus", forestci=False, image=True, csv=True): #define function that trains up to eight models at once plots with each model in a subplot. Includes model scores
    global train_data, train_data, test_data, test_target #we need these variables and don't want to pass them as arguments
    with plt.rc_context({'xtick.color':'white', 'ytick.color':'white','axes.titlecolor':'white','figure.facecolor':'#1e1e1e','text.color':'white','legend.labelcolor':'black'}): #use (1, 1, 1, 0) for figure.facecolor for transparent bg
        warnings.filterwarnings("ignore")
        fig, ax = plt.subplots(2, 4, sharey='row', figsize=(28,10))
        fig.subplots_adjust(hspace=0.35)
        fig.suptitle(title, color='white', size=16)
        results = pd.DataFrame(columns=("model","mean_squared_error","mean_absolute_error","max_error","r2_score","mapie_eff_mean_width"))

        for y, col in enumerate(models):
            for x, [model_name, model, parameters, uncert] in enumerate(col):
                regressor = model(**parameters)
                if uncert and not forestci and method != "prefit": #error calculations need magie training if not forestci/prefit mapie
                    mapie_regressor = MapieRegressor(estimator=regressor, method=method) #unpacks model and params
                    if model_name in ("Superlearner", "Random Forest - Lolopy"): #need to get values for these models
                        mapie_regressor.fit(train_data.values, train_target.values) #fit the model
                    else:
                        mapie_regressor.fit(train_data, train_target) #fit the model
                        model_pred, model_pis = mapie_regressor.predict(test_data, alpha=0.05) #make predictions on test data
                else: #no need for error calculations during training, use sklearn
                    if model_name in ("Superlearner", "Random Forest - Lolopy"): #need to get values for these models
                        regressor.fit(train_data.values, train_target.values) #fit the model
                    else:
                        regressor.fit(train_data, train_target) #fit the model
                    model_pred = regressor.predict(test_data) #make predictions on test data

                mws = None #set mean width score as null until mapie if statement
                mse = round(mean_squared_error(test_target, model_pred),3) #find mean square error
                mae = round(mean_absolute_error(test_target, model_pred),3) #find mean square error
                mxe = round(max_error(test_target, model_pred),3)
                r2 = round(r2_score(test_target, model_pred),3) #find r2 score 


                #make our plot - with plt.rc_context sets theme to look good in dark mode
                difference = np.abs(test_target - model_pred) #function that finds the absolute difference between predicted and actual value
                im = ax[x, y].scatter(test_target, model_pred, cmap='plasma_r', norm=plt.Normalize(0, 120), c=difference, label="Critical Temperature (K)", zorder=2) #create scatter plot of data 
                ax[x, y].plot((0,135), (0,135), 'k--', alpha=0.75, zorder=3) #add expected line. Values must be changed with different data to look good

                if uncert: #plot error bars
                    if forestci and model is RandomForestRegressor:
                        model_unbiased = fci.random_forest_error(regressor, train_data, test_data, calibrate=False) #NOTE: forestci calibration is disabled as there is a bug in the code (they use a too small datatype)
                        yerror = np.sqrt(model_unbiased)
                    elif method == "prefit":
                        raise NameError("Prefit method is not implemented in this program as our current implementation have error bars that do not align with the test data")
                        # mapie = MapieRegressor(estimator=model, cv="prefit").fit(cal_data, cal_target) #important: calibration data must be different from training data!
                        # pred_interval = pd.DataFrame(mapie.predict(test_data, alpha=.05)[1].reshape(-1,2), index=test_data.index, columns=["lower", "upper"]) #get interval predictions on test data, with alpha=5%
                        # yerror = pred_interval.values.reshape(2,-1)
                    else:
                        #model_pis contains absolute points for upper/lower bounds. We need absolute error, like (3, 3) for ± 3:
                        yerror = np.abs(model_pis[:,:,0].transpose() - np.tile(model_pred, (2, 1))) #error must be in shape (n, 2) for errorbars
                        mws = round(regression_mws(model_pis[:,:,0][:,0],model_pis[:,:,0][:,1]),3) #generate mean width score metric from mapie data
                    ax[x, y].errorbar(test_target, model_pred, yerr=yerror, fmt="none", ecolor="black", alpha=0.5, zorder=1) #removed ", label="Prediction Intervals", as it is covers other data for bulk plots 

                results.loc[len(results.index)] = (model_name,mse,mae,mxe,r2,mws)

                ax[x, y].set_title(model_name, c='white')
                ax[x, y].set_ylabel('Prediction', c='white')
                ax[x, y].set_xlabel('Actual Value', c='white')
                ax[x, y].annotate(f'R2: {r2}', xy = (0, -0.15), xycoords='axes fraction', ha='left', va="center", fontsize=10) #add footnote with R2 
                ax[x, y].annotate(f'MWS: {mws}' if mws else f'MXE: {mxe}', xy = (0, -0.20), xycoords='axes fraction', ha='left', va="center", fontsize=10) #add footnote with R2 
                ax[x, y].annotate(f'MAE: {mae}', xy = (1.0, -0.20), xycoords='axes fraction', ha='right', va="center", fontsize=10) #add footnote with MAE
                ax[x, y].annotate(f'MSE: {mse}', xy = (1.0, -0.15), xycoords='axes fraction', ha='right', va="center", fontsize=10) #add footnote with MSE

        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles=handles,loc='lower center')

        fig.colorbar(im, ax=ax.ravel().tolist()).set_label(label="Difference from Actual (K)", color='white') #using .set_label() as colorbar() does accept color arguments
        if image:
            plt.savefig(f'../data/{filename}.png', bbox_inches='tight')
        if csv:
            results.to_csv(f'../data/{filename}.csv', index=False)
        plt.show()