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

## Thursday, June 20th, 2022
* **Practiced presentation**
* **Completed REU interview Q&A**

</details>
<br>
