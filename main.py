import numpy as np
import pandas as pd
import os
import matplotlib as plt
import time

from preprocessing import *
from models.decision_tree import *
from models.naive_bayes import *
from models.kNN import *
from evaluate import *

pd.options.mode.chained_assignment = None # disable SettingWithCopyWarning
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)}) # format np array prints to 2 decimal places

iris_data = iris_data = pd.read_csv('./input/iris.csv')
shuffled_data = iris_data.sample(frac=1) # shuffle data

k=10 # param for k-fold-cross-validation, commonly 10
model_mapping = {0: 'Decision Tree', 1: 'Naive Bayes', 2: 'k-nearest neighbor'}
accuracy = np.zeros((k, len(model_mapping)))

start = time.time() # TODO: function specific timer would be neat
#-----------------------------------------------------------
# k-fold cross-validation
for i in range(k) : 
    trainset, testset = get_train_test_split(i, k, shuffled_data)

    ##--------------- decision tree ----------------------
    root = decision_tree(trainset, verbose=False)
    accuracy[i, 0] = evaluate_model(root, testset)
    #print(f'Decision tree representation in breadth-first traversal: {root}')

    ##--------------- naive bayes ----------------------
    bayes_model = naive_bayes_model(shuffled_data)
    accuracy[i, 1] = evaluate_model(bayes_model, testset)

    ##--------------- k-nearest neighbor ----------------------
    kNN = kNN_model(shuffled_data, k=3)
    accuracy[i, 2] = evaluate_model(kNN, testset)
    
    #print(f'Accuracy for iteration {i}: {accuracy[i]}')

avg_model_accuracy = np.average(accuracy, axis=0)
end = time.time()

for i in range(len(model_mapping)) :
    print(f'Average model accuracy for {model_mapping.get(i)}: {avg_model_accuracy[i]:.2f}')
print(f'Total execution time: {end - start:.3f} seconds')



