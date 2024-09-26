import numpy as np
import pandas as pd
import os
import matplotlib as plt

from preprocessing import *
from decision_tree import *

iris_data = iris_data = pd.read_csv('./input/iris.csv')

#-----------------------------------------------------------
# k-fold cross-validation
k=10
i = 3

shuffled_data = iris_data.sample(frac=1) # shuffle data
trainset, testset = get_train_test_split(i, k, shuffled_data)
#----------------------------------------------------------

##--------------- decision tree model building ----------------------

   # trainset = trainset[trainset['variety'] != variety]
#trainset_left = trainset[trainset['lala'] < 5]
root = build_decision_tree(trainset, verbose=True)

# TODO add function to traverse tree for data point classification
# TODO evaluate model accuracy against testset

#print(root.traverse(testset.iloc[0]))
#print(testset.iloc[0])

#print(testset.iloc[0].variety)

#print(testset.iloc[0]['sepal.length'])

print(root.show_tree())
#print('----------------')
#print(root.left.show_tree())
#print('------------------')
#print(root.right.show_tree())

#    tree_node 
#L:Set  --   tree_node
#        tree_node --- L:Virg
#    L:Vers -- ERROR
    



