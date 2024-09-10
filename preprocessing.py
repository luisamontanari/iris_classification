import math
import pandas as pd

# TODO implement actual k fold cross-validation

def get_train_test_split(i, k, shuffled_data) : 
    step_size = math.floor(shuffled_data.shape[0]/k)
    testset = shuffled_data[i*step_size : i*step_size+step_size]

    # TODO fix the index problem here
    trainset = pd.concat([shuffled_data[:i*step_size-1], shuffled_data[i*step_size+step_size-1:]])
    return trainset, testset