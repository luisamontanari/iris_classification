import numpy as np
import pandas as pd
import os
import matplotlib as plt
import time

from preprocessing import *
from decision_tree import *
from evaluate import *

pd.options.mode.chained_assignment = None # disable SettingWithCopyWarning

iris_data = iris_data = pd.read_csv('./input/iris.csv')
shuffled_data = iris_data.sample(frac=1) # shuffle data

k=10 # param for k-fold-cross-validation, commonly 10
accuracy = np.zeros(k)

start = time.time() # TODO: function specific timer would be neat
#-----------------------------------------------------------
# k-fold cross-validation
for i in range(k) : 
    trainset, testset = get_train_test_split(i, k, shuffled_data)
    #----------------------------------------------------------
    
    ##--------------- decision tree model building ----------------------

    root = build_decision_tree(trainset, verbose=False)

    #print(f'Prediction for dataset head:\n{root.classify_dataset(testset.head())}')

    accuracy[i] = evaluate_model(root, testset)
    #print(f'Accuracy for iteration {i}: {accuracy[i]:.2f}')
    print(root)

avg_model_accuracy = np.average(accuracy)
end = time.time() #11.11.24: avg time 4.9 seconds

print(f'Average model accuracy for decision tree: {avg_model_accuracy:.2f}')
print(f'Total execution time: {end - start:.3f} seconds')



