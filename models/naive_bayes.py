import pandas as pd
import numpy as np
from models.ml_model import *

class naive_bayes_model(ml_model):
    # Naive Bayes assumption: all features are independent
    # assumption is rarely true, but NB is still surprisingly accurate in classification tasks (but not regression!)
    def __init__(self, training_data : pd.core.frame.DataFrame) :
        variance_mat, means_mat, priors_vec, class_mapping = self._build_naive_bayes_model(training_data)
        
        # shape of all matrices: (#classes, #features), i.e. (3, 4)
        self.variance_mat = variance_mat.astype(float)
        self.means_mat = means_mat.astype(float)
        self.priors = priors_vec.astype(float)
        self.class_mapping = class_mapping # map class indices to labels
    
    def classify_datapoint(self, datapoint : pd.Series) -> str :
        dp = datapoint.drop(labels=['variety']).to_numpy().flatten().astype(float)
        
        # P(x(i) | y), likelihood of our observation of feature x(i) (columns) assuming class x is present (rows)
        likelihood_mat = 1/np.sqrt(2*np.pi*self.variance_mat) * np.exp(-((dp-self.means_mat)**2) / (2*self.variance_mat)) # Gaussian PDF
        
        # prediction(X) 
        # = argmax_y P(y | X)  ---> we predict the maximally likely class y given our observation X
        # = argmax_y P(y) * prod([ P(x(i) | y) for each i in n ])
        class_probabilities = self.priors * np.prod(likelihood_mat, axis=1)
        prediction = self.class_mapping.get(np.argmax(class_probabilities))
        return prediction
    
    def _build_naive_bayes_model(self, df : pd.core.frame.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict] : 
        # model building: get priors, mean and variance matrix
        feature_count = df.shape[1]-1
        class_count = df['variety'].unique().shape[0]
        variance_mat = np.zeros((class_count, feature_count))
        means_mat = np.zeros((class_count, feature_count))
        priors_vec = np.zeros((class_count,))
        class_mapping = dict() # to keep track of which class label is associated with which row index
        
        for i, variety in enumerate(df['variety'].unique()):
            class_mapping.update({i:variety})
            # segment dataframe by variety and convert to numpy array
            df_by_variety = df.loc[df['variety'] == variety]
            obs = df_by_variety.drop('variety', axis=1).to_numpy()
            n = obs.shape[0]

            # calc sample class mean, variance and prior per variety
            mean_arr = np.mean(obs, axis=0)
            means_mat[i,:] = mean_arr
            
            prior = obs.shape[0]/df.shape[0] # relative frequency in dataset
            priors_vec[i] = prior
            
            if (n == 1): continue # TODO: implement actual try catch block
            # calc Bessel-corrected sample class variance (Bessel-correction removes bias introduced from calculating mean and variance from the same sample)
            variance_arr = 1.0/(n-1) * np.sum((obs-mean_arr)**2, axis=0)
            variance_mat[i,:] = variance_arr
        return variance_mat, means_mat, priors_vec, class_mapping
