import numpy as np
import pandas as pd
import os
import matplotlib as plt

from preprocessing import *
from models.decision_tree import *
from evaluate import *
from models.naive_bayes import *
from models.kNN import *

iris_data = iris_data = pd.read_csv('./input/iris.csv')
shuffled_data = iris_data.sample(frac=1) # shuffle data
print(shuffled_data.head()['variety'])
print('------------------')
kNN = kNN_model(shuffled_data.head(), 3)
print(kNN.labels)
print(kNN.data)
shuffled_data.head(1).apply(kNN.classify_datapoint, axis=1)