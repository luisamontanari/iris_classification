# Iris classification

This repository contains a small personal project in which I build classification models using the famous [iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set). The following models were implemented: 

- Decision tree
- Gaussian Naive Bayes
- k-nearest neighbor

I've opted not to use any existing implemententions of the above models such as those provided by the packages like `scikit`. Instead, I implemented all models from scratch to better understand what's going on behind the scenes. Here I relied heavily on the `pandas` and `numpy` packages.
Using k-fold-classification, each model's performance is trained and evaluated on different dataset training and testing splits. Naive Bayes emerges as the most accurate classifier on this dataset. 

The code can be exeuted by running `python main.py`. Options `v` (verbose) and/or `d` (debug) can be added to display additional information. 
