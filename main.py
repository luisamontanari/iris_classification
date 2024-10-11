import numpy as np
import pandas as pd
import os
import matplotlib as plt

from preprocessing import *
from decision_tree import *
from evaluate import *

pd.options.mode.chained_assignment = None # disable SettingWithCopyWarning

iris_data = iris_data = pd.read_csv('./input/iris.csv')

#-----------------------------------------------------------
# k-fold cross-validation
k=10
i = 3

shuffled_data = iris_data.sample(frac=1) # shuffle data
trainset, testset = get_train_test_split(i, k, shuffled_data)
#----------------------------------------------------------

# TODO: add timer for execution

##--------------- decision tree model building ----------------------

root = build_decision_tree(trainset, verbose=False)

print(f'Prediction for dataset head:\n{root.classify_dataset(testset.head())}')

print(f'Accuracy: {evaluate(root, testset)}')

print(root.show_tree())




