import pandas as pd
import math
import numpy as np

class naive_bayes_model:
    # Naive Bayes assumption: all features are independent
    # assumption is rarely true, but NB is still surprisingly accurate in classification tasks (but not regression!)
    def __init__(self, training_data):
        variance_mat, means_mat, priors_vec, class_mapping = self._build_naive_bayes_model(training_data)
        
        # shape of all matrices: (#classes, #features), i.e. (3, 4)
        self.variance_mat = variance_mat.astype(float)
        self.means_mat = means_mat.astype(float)
        self.priors = priors_vec.astype(float)
        self.class_mapping = class_mapping # map class indices to labels
        return
    
    # classify a single datapoint
    def classify_datapoint(self, datapoint):
        dp = datapoint.drop(labels=['variety']).to_numpy().flatten().astype(float)
        
        # P(x(i) | y), likelihood of our observation of feature x(i) (columns) assuming class x is present (rows)
        likelihood_mat = 1/np.sqrt(2*np.pi*self.variance_mat) * np.exp(-((dp-self.means_mat)**2) / (2*self.variance_mat)) # Gaussian PDF
        
        # prediction(X) 
        # = argmax_y P(y | X)
        # = argmax_y P(y) * prod([ P(x(i) | y) for each i in n ])
        class_probabilities = self.priors * np.prod(likelihood_mat, axis=1)
        prediction = self.class_mapping.get(np.argmax(class_probabilities))
        return prediction
    
    # classify a set of datapoints
    def classify_dataset(self, df) :
        df.loc[:, 'classification'] = df.apply(self.classify_datapoint, axis=1, result_type='expand')
        return df


# Naive Bayes Assumption: 


# formula (assuming a Gaussian distribution)
# X = x(1) ... x(n) is our observation vector

# our classification of X will be the maximum likelihood estimator for P(y | X) over all possible classes y

# prediction(X) 
# = argmax_y P(y | X)
# = argmax_y P(y) * prod([ P(x(i) | y) for each i in n ])
# (the last inequality hinges on our naive bayes assumptions!)

# for the prior P(y) we use the relative frequency of y in dataset
# for P(x(i) | y) we assume an underlying Gaussian distribution
# P(x(i) | y) 
# = 1/np.sqrt(2*np.pi*variance_y_i)*np.exp(-(x(i)-mean_y_i)**2/2*variance)
# where 
# variance_y_i -> Bessel-corrected empirical variance of x(i) associated with class y
# = sum_i_in_n((x(i) - mean_y_i)**2) * 1/(n-1) 
# mean_y_i -> empirical mean of x(i) associated with class y
# = sum(x_y_i)/count(x_y_i) where x_y_i = set of values for x(i) when restricted to class y

    def _build_naive_bayes_model(self, df) : 
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

            # calc sample class mean, variance and prior
            mean_arr = np.mean(obs, axis=0)
            means_mat[i,:] = mean_arr
            
            prior = obs.shape[0]/df.shape[0] # relative frequency in dataset
            priors_vec[i] = prior
            
            if (n == 1): continue # TODO: implement actual try catch block
            # calc Bessel-corrected sample class variance
            variance_arr = 1.0/(n-1) * np.sum((obs-mean_arr)**2, axis=0)
            variance_mat[i,:] = variance_arr
        return variance_mat, means_mat, priors_vec, class_mapping
