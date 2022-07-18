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