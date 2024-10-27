import numpy as np
import pandas as pd
import os
import matplotlib as plt

from preprocessing import *
from decision_tree import *
from evaluate import *
from naive_bayes import *

iris_data = iris_data = pd.read_csv('./input/iris.csv')
shuffled_data = iris_data.sample(frac=1) # shuffle data
print(shuffled_data.head(10))
print('------------------')
bayes_model = naive_bayes_model(shuffled_data)
#bayes_model.classify_datapoint(shuffled_data.head(1))
print(bayes_model.classify_dataset(shuffled_data))


vari = np.array([0.4, 0.1, 0.3, 0.07])
print(type(vari), type(shuffled_data))

means = np.array([6.59, 2.97, 5.55, 2.026])
obs = np.array([6.7, 3.3, 5.7, 2.5])
f1 = 1/(np.sqrt(2*np.pi*vari))
f2 = np.exp(-((obs-means)**2) / (2*vari))
f3 = f1 * f2
#print(f3)