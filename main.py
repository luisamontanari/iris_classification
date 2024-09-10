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

def build_decision_tree(trainset, g_i=1, node=None, isUnderThreshold=False, variety=None) :

    # TODO: add Rekursionanker with a sensible condition (if GI is low or the number of people is low or the tree is too big  etc)
    if g_i < 0.3 :
        # TODO I don't think we actually need to remember the variety, we can just take the variety with the most data point in the result
        # maybe we don't need the direction either?  Think this thorugh
        if isUnderThreshold : node.left = decision_tree(leaf(variety))
        else : node.right = decision_tree(leaf(variety))
        return node

    g_i, threshold, isUnderThreshold, feature, variety = get_purest_node(trainset, verbose=True)

    node = decision_tree(tree_node(threshold, feature))

    # recursively calc purest node on both child segmentions: 
    trainset_left = trainset[trainset[feature] < threshold]
    trainset_right = trainset[trainset[feature] >= threshold]

    build_decision_tree(trainset_left, g_i, node, isUnderThreshold, variety)
    build_decision_tree(trainset_right, g_i, node, isUnderThreshold, variety)

    return node

   # trainset = trainset[trainset['variety'] != variety]

root = build_decision_tree(trainset)

# TODO add function to traverse tree for data point classification
# TODO evaluate model accuracy against testset


