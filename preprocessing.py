import math
import pandas as pd

# TODO implement actual k fold cross-validation

def get_train_test_split(i, k, shuffled_data) : 
    step_size = math.floor(shuffled_data.shape[0]/k)
    start_idx = i*step_size
    end_idx = i*step_size+step_size
    testset = shuffled_data[start_idx : end_idx-1]
    trainset = shuffled_data.drop(range(start_idx, end_idx), axis=0)
    return trainset, testset