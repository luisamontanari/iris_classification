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
print(shuffled_data.head())
print('------------------')
bayes(shuffled_data.head())