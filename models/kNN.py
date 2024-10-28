import pandas as pd
import numpy as np
from models.ml_model import *

class kNN_model(ml_model):
    # k-nearest neighbor model
    def __init__(self, training_data : pd.core.frame.DataFrame, k : int) :
        self.labels = training_data['variety'].to_numpy()
        self.data = training_data.drop(columns=['variety']).to_numpy()
        self.k = k
        return
    
    def classify_datapoint(self, datapoint : pd.Series) -> str :
        dp = datapoint.drop(labels=['variety']).to_numpy().flatten().astype(float)
        euclid_dist_vec = np.sum((self.data - dp)**2, axis=1)
        k_nearest_neighbors = np.argpartition(euclid_dist_vec, self.k)[:self.k] # index array of k entries with smallest distance
        # get most frequent label among the k nearest neighbors
        k_nearest_labels = self.labels[k_nearest_neighbors]
        u, indices = np.unique(k_nearest_labels, return_inverse=True)
        prediction = u[np.argmax(np.bincount(indices))]
        return prediction