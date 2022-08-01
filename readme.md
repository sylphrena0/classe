# **Cornell CLASSE REU Repository**
*Author/Mentee: Kirk Kleinsasser*

*Mentor: Suchismita Sarker*

This repository contains code, documentation, and data produced during my 2022 REU program at Cornell. Details on the REU program can be found [here](https://www.classe.cornell.edu/StudentOpportunities/ReuProgram.html).

This program uses [matminer](https://hackingmaterials.lbl.gov/matminer/) to extract data from the [supercon database](https://supercon.nims.go.jp/), extracted from [this github repository](https://github.com/vstanev1/Supercon). This extracted data is then used to train machine learning algorithms to predict critical tempertures of superconducters based on their features.


**Dependancies:**
 - [pandas](https://pypi.org/project/pandas/)
 - [numpy](https://pypi.org/project/numpy/)
 - [seaborn](https://pypi.org/project/seaborn/)
 - [matplotlib](https://pypi.org/project/matplotlib/)
 - [matminer](https://pypi.org/project/matminer/)
 - [scikit-learn](https://pypi.org/project/scikit-learn/)
 - [scikit-optimize](https://pypi.org/project/scikit-optimize/)
 - [mlens](https://pypi.org/project/mlens/)
 - [juypter-notebook](https://jupyter.org/)

<details open>
<summary><font size=4>Project Filestructure:</font></summary>

 ```
root
│   .bashrc             ~ bash functions to simplify workflow
│   .gitignore          ~ specifies files that git will not send to this github repository (mostly runtime files)
│   compute.sh          ~ file like run_script.sh that attempts to add bash arguments to control qsub
│   compute_test.sh     ~ alternative attempt to add bash arguments
│   documentation.md    ~ documents weekly work during the CLASSE REU program
│   readme.md           ~ explains project and script dependancies - you are here!
│   run_script.sh       ~ runs any python script in qsub. Run (using .bashrc) with syntax "qsub <script>.py <args>" 
│
└───code    ~ contains all code for the project, excluding bash scripts
│   │   feature_anaylsis.ipynb      ~ anaylzes features and generates files that give a landscape of that database
│   │   feature_selection.ipynb     ~ anaylzes feature importance and correlations
│   │   training.ipynb              ~ training notebook to test models locally before running compute farm scripts
│   │   build_features.py           ~ extracts features from datasets with matminer
│   │   model_optimizer.py          ~ optimizes sklearn models with GridSearchCV
│   │   model_optimizer_bayes.py    ~ optimizes sklearn models using bayesian optimization with scikit-optimize
│   │   training_bulk.py            ~ trains up to eight models at once to generate a combined result graph and csv
│   │   training_single.py          ~ trains single models and can export feature importances and graphs
│   │
│   └───dependancies    ~ contains code that defines shared functions, used by code in the parent directory
│       │   shared_functions.py     ~ general use functions that are used in many files
│       │   superlearner.py         ~ functions that simplify creation of superlearning models
│       │   ...
│
└───data    ~  contains datasets, features, and various generated files about the data - feature files include target
│   │   dataset.csv                 ~ superconducter database from vstanev1/Supercon
│   │   dataset_no_outliers.csv     ~ superconductor database, removing outliers from the data
│   │   features.csv                ~ features for training, generated from dataset with ../code/build_features.py
│   │   features_no_outliers.csv    ~ features for training using data without outliers
│   │   dataset_histogram.png       ~ histogram of critical tempurtures in the dataset
│   │   feature_heatmap.png         ~ heatmap of the correlations between features and the target
│   │   feature_histograms.png      ~ histograms of all the features in the data
│   │
│   └───importance    ~ contains feature importances for ensemble models, generated by code/feature_selection.ipynb
│       │   ...
│
└───latex   ~ contains source files and output for the latex final paper for the CLASSE REU program
│   ...
│
└───results     ~ contains all result prediction vs target grapgs and exported files
    │   results_optimized.csv               ~ results from training main eight models, generated from ../code/training_bulk.py
    │   results_unoptimized.csv             ~ unoptimized results from training main eight models
    │   results_optimized.png               ~ graph of results of main eight models
    │   results_unoptimized.png             ~ graph of unoptimized results of main eight models
    │   results_unoptimized_optimized.png   ~ graph of four unoptimized results vs optimized results
    │
    └───individual      ~ contains graphs and csv result files from individual model training 
    │   │   ...
    │
    └───optimization    ~ contains csv results from code/model_optimizer.py and code/model_optimizer_bayes.py
        │   ...
```

</details>
<br>