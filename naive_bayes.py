import pandas as pd
import math
import numpy as np

class naive_bayes_model:
    def __init__(self, data):
        pass
    
    # classify a single datapoint
    def classify_datapoint(self, datapoint):
        pass
    
    # classify a set of datapoints
    def classify_dataset(self, df) :
        pass


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

def bayes(df) : 
    # -------------------- model building: get priors, mean and variance matrix
    # s.l  s.w  p.l  p.w
    # 5.1, 3.5, 1.4, .2
    # 4.9, 3,   1.4, .2
    # 4.7, 3.2, 1.3, .2
    for variety in df['variety'].unique(): # this too can be vectorized -> get mean/variance matrices instead of vectors
        # segment dataframe by variety and convert to numpy array
        df_by_variety = df.loc[df['variety'] == variety]
        obs = df_by_variety.drop('variety', axis=1).to_numpy()
        n = obs.shape[0]

        # calc sample mean, variance and prior
        mean_arr = np.mean(obs, axis=0)
        prior = obs.shape[0]/df.shape[0]

        #print(obs)
        #print(mean_arr)
        #print(np.sum((obs-mean_arr)**2, axis=0))
        print('_____')
        if (n == 1): continue # TODO: implement actual try catch block
        # calc Bessel-corrected sample variance
        variance_arr = 1.0/(n-1) * np.sum((obs-mean_arr)**2, axis=0)

        # ----------- i think actual model building is done here, from now we assume there is an observation
        
        ## P(x(i) | y) 
        # = 1/np.sqrt(2*np.pi*variance_y_i)*np.exp(-(x(i)-mean_y_i)**2/2*variance)
        likelihood_arr = 1/np.sqrt(2*np.pi*variance_arr) * np.exp(-(obs-mean_arr)**2/2*variance_arr) # TODO: I think something is wrong here...
        print(likelihood_arr)

        # prediction(X) 
        # = argmax_y P(y | X)
        # = argmax_y P(y) * prod([ P(x(i) | y) for each i in n ])

    return
