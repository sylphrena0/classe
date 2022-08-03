# **REU Documentation**
Documents daily work in the REU program, including events, meetings, and broad descriptions of programming work.

*Author: Kirk Kleinsasser*

<br>

<details>
<summary><font size=5>Week 1</font></summary>

---

## Tuesday, June 7th, 2022

* **Installed matminer package**
* **Conducted some background research into superconducters**
* **Attended the CHESS user meeting and lab training**

## Wednesday, June 8th, 2022
* **Attended Wilson safety training**
* **Created introduction to superconducters presentation slide**
* **Searched for materials database**
1. [Supercon Database](https://en.iric.imet-db.ru/DBinfo.asp?idd=51)
2. [ICSD Database](https://icsd.products.fiz-karlsruhe.de/) (not just superconducters, the world's largest database for completely identified inorganic crystal structures )

## Thursday, June 9th, 2022
* **Took online safety trainings**
* **Looked into matminer documentation**

## Friday, June 10th, 2022
* **Imposter syndrome workshop!**
* **Started testing matminer featurizers and automatminer**
* **Worked on python package conflicts with automatminer (issue with sklearn depreciating a metric import)**
</details>
<br>


<details>
<summary><font size=5>Week 2</font></summary>

---

## Monday, June 13th, 2022
* **Intro to Accelerator Science Workshop**
* **REU weekly meeting**
* **CLASSE 998 Safety Training**
* **Abandon automatminer for a manual implementation due to unresolved package errors**
* **Worked more on testing matminer featurizers, still having some issues getting proper composition data.**

## Tuesday, June 14th, 2022
* **Library information session**
* **Composition issues fixed, featurizers working correctly**
* **Testing some machine learning algorithms**
* **Looking into the compute farm, as some of the featurizers have time estimates higher than an hour**

## Wednesday, June 15th, 2022
* **Ethics in science session**
* **Featurizering complete**
* **Started machine learning notebook**
* **Organized code and added readme**

## Thursday, June 16th, 2022
* **Worked on moving residence halls**
* **Futher work on regression models and gridsearch techniques**
* **Optimized nonlinear SVR model**
* **Added more comments to code**
* **Reading up on superconducters while code runs**

## Friday, June 17th, 2022
* **Added slide to my [presentation](https://docs.google.com/presentation/d/1c-wVKFG8I19eHjtP6NVvT8W4JUKoA0xY--zjsJw1pV4/edit#slide=id.g13564193239_0_6)**
* **Setup optimization for all remaining models and sent to compute farm to run overnight.**
</details>
<br>

<details>
<summary><font size=5>Week 3</font></summary>

---

## Tuesday, June 21st, 2022
* **Previous optimization run failed over the weekend - added error handling and multiprocessing to speed up script.**
* **Continued work on presentation slides**
* **Looked into compute farm syntax to optimize code. The compute farm is easier - while I can run it on my laptop, I need to turn off my laptop to switch operating systems at night, plus I can enable email notifications for scripts on the compute farm.**
* **Added script that runs optimizers on compute farm instead of manually running code**

## Wednesday, June 22nd, 2022
* **REU weekly meeting**
* **Continued work on presentation and optimization**
* **Calculated complexity of calculations**
* **Submitted a second optimizer that only optimizes a subset of the dataset to save on computations - will use if the other computation takes too long**

## Thursday, June 23rd, 2022
* **Overhauled optimization scripts - now accepts sample size argument and can disable overly computationaly heavy models (ahem SVMs). This vastly improves optimization and should make future optimization much easier. Our best R2 value to date is around .80, with a mean squared error of aroun 140.**

## Friday, June 24th, 2022
* **Made matplotlib function that plots actual vs. predicted values. Added plots for predicted vs actual values. Configured annotations to show metrics and added heatmap to show distance from target.**
* **Made second matplotlib function that plots up to eight models at a time for easy comparision with all previously mentioned features.**
* **Reoganized and better documented purpose of files in /code/.**
</details>
<br>

<details>
<summary><font size=5>Week 4</font></summary>

---

## Monday, June 27th, 2022
* **Introduction to Accelerator Physics Talk**
* **Weekly REU meeting**
* **Worked on matplotlib graphs - adding y=x line to subplots to show the optimial results.**
* **Fixed mislabeled axises.**

## Tuesday, June 28th, 2022
* **Worked on presentation**
* **Added MAE and Max Error (max error is mainly to make metrics symmetrical on graph)**
* **Started running of superlearner model to add to graphs**

## Wednesday, June 29th, 2022
* **Finished presentation slides**
* **Enabled superlearners results in compute functions**
* **Added optimized results to presentation**
* **Started to practice presentation**

## Thursday, June 30th, 2022
* **Practiced presentation**
* **Completed REU interview Q&A**

## Friday, July 1st, 2022
* **Presented for other REU students**

</details>
<br>

<details>
<summary><font size=5>Week 5</font></summary>

---

## Tuesday, July 5th, 2022
* **Started looking into lolopy and uncertainty calculations**
* **Started latex document (trouble with configuration)**
* **Did some code organization**

## Wednesday, July 6th, 2022
* **Switched compute farm scripts to use user installed python packages**
* **Sent lolopy training to compute farm due to computational cost of training model (upwards of two hours)**
* **Sent remaining optimization script to compute farm**

## Thursday, July 7th, 2022
* **Discovered and fixed bugs in both compute farm scripts**
* **Added quality of life bash functions to speed up compute farm work**
* **Found much better RFR (same metrics, much better certainty)**
* **Overhauled compute scripts and crushed bugs**

## Friday, July 8th, 2022
* **Moved functions that are shared between files to ./dependancies and imported them in relevent files to limit redundancy. This also means we have less variables stored in each file.**
* **Looked into [mapie](https://mapie.readthedocs.io/en/latest/tutorial_regression.html) and other uncertainty measures to potentially add a model agnostic uncertainty measure**
* **Attended graduate school panel**


</details>
<br>

<details>
<summary><font size=5>Week 6</font></summary>

---

## Monday, July 11th, 2022
* **Research into mapie, implemented into an errorbar. Considered just adding a metric, tryed to replicate results from forestci.**
* **Updated feaurizer scripts and submitted compute job to generate features for simulated dataset.**

## Tuesday, July 12th, 2022
* **Implemented mapie with errorbars that work on any model**
* **Modified functions to accept mapie arguments**

## Wednesday, July 13th, 2022
* **Worked on featurizing simulated dataset, squished some bugs**
* **Started to draft paper - superconducters/matminer/ml background**
* **Continued to try lolopy training - compute farm**

## Thursday, July 14th, 2022
* **Featurized failed to export data after running for <5 hours, reran with updated export command**
* **Successfully featurized simulated data, ready to train/apply models**
* **Continued to draft paper - superconducters/matminer**
* **Ran into another lolopy issue, may have traced the issue to the source - one of the dependancies might not be able to be installed with `pip3 install--user`**

## Friday, July 15th, 2022
* **Lolopy ran out of memory, reran with more allocated**
* **Worked on methodlogy in the REU report, started results section**
* **Ran simulated model training (single), results in ./data/individual/**
* **Tried to troubleshoot lens superlearner error with mapie so we can get bulk_results (we can now, but not with error with superlearner)**
* **Added a bulk training script and updated bulk training function**

</details>
<br>

<details>
<summary><font size=5>Week 7</font></summary>

---

## Monday, July 18th, 2022
* **All lolopy jobs failed over the weekend due to lack of memory - try again with [higher java memory allocation](https://github.com/CitrineInformatics/lolo/tree/main/python#use)**
* **Continue to document work in latex report**
* **Attended weekly meeting and accelerator physics talk, toured Newman facilities**
* **Got SVR results with error! Still waiting on superlearner results**
* **Tried some basic feature selection (dropped low correlated data >0.05)**

## Tuesday, July 19th, 2022
* **Lolopy jobs are still not complete. Started unoptimized bulk training with error**
* **Meeting/chat with Rick to discuss research and take photos**
* **Everything is working except the simulated superlearner**

## Wednesday, July 20th, 2022
* **Tour of Wilson Lab with Susan Newman**
* **Medical Appointment**

## Thursday, July 21th, 2022
* **Workshop on Scientific Presentation Skills**
* **Fixed bug in bulk training for unoptimized models**
* **Updated scripts to output to a single file**
* **Did feature selection on best models**

## Friday, July 21th, 2022
* **Workshop on Scientific Paper Skills**
* **Fixed incorrect name for dataset w/o outliers (simulated --> outliers)**
* **Started rerun of outlier dataset with correct labels.**
* **Added option to run all models in single script**

</details>
<br>

<details>
<summary><font size=5>Week 8</font></summary>

---

## Monday, July 25th, 2022
* **Worked on feature anaylsis, produced histogram of target**
* **Added ability to drop data above/below certain temps from data import**
* **Trained random forest models with cutoff critical temps. MAE is smaller on model using <10K (more data in that set)**

## Tuesday, July 26th, 2022
* **Attended machine learning conference**
* **Added option to export csv with result metrics to both evalation functions to easily make tables later this week**
* **Added mean width score from mapie for mapie trained models**
* **Reran bulk training on compute farm to get updated data**

## Wednesday, July 27th, 2022
* **Attended machine learning conference**
* **Lolopy - ran multiple times, increasing memory limit up to 32GBs, still running out of memory. Could use lolo directly or downsize training size, but it might be worth skipping as both options would likely take significant time.**
* **Started working on ./code/model_optimizer_bayes so we can optimize with acq fncts.**

## Thursday, July 28th, 2022
* **Continued working on ./code/model_optimizer_bayes (fixing output).**
* **Continued working on latex paper.**

## Friday, July 29th, 2022
* **Worked on generalizing bash scripts - I kept hitting roadblocks. These updated scripts still don't really work, I will leave them be until I have more time.**
* **I did get one version of a generalized bash script to work, it just doesn't accept arguments in bash (but it works well). Updated scripts to use that and kept the working file if I have time to improve more.**
* **Ran more bayesian optimization, will examine results on monday (train models and maybe get new optimial model). Also need to do more models with selected features on monday.**
* **Ran limited Tc models with no outliers.**
* **Met with a staff scientist to chat about career stuff!**

</details>
<br>

<details>
<summary><font size=5>Week 9</font></summary>

---

## Monday, July 25th, 2022
* **REU Weekly Meeting**
* **Trained models based on bayesian optimization (in [code/training.ipynb](code/training.ipynb))**
* **Worked on presentation and paper**
* **Restructed code (moved results to seperate folder, renamed some files)**
* **Added explaination of filestructure and purpose of all files**

## Tuesday, July 26th, 2022
* **Further work on presentation and powerpoint (mainly discussion of optimization)**
* **Fixed random state to normalize results, reran training**

</details>
<br>