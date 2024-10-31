from models.ml_model import *
import math

# evaluate model on the testset and output its accuracy
def evaluate_model(model : ml_model, dataset : pd.DataFrame) -> int:
    prediction = model.classify_dataset(dataset)
    merged_df = pd.concat([dataset, prediction], axis=1)
    correct_classifications = merged_df.apply(lambda dp : int(dp['variety'] == dp['prediction']), axis=1)
    accuracy = correct_classifications.sum()/dataset.shape[0]
    return accuracy

# split training data into k segments, return the ith segment as a testset and the rest as trainset
def get_train_test_split(shuffled_data : pd.DataFrame, i : int = 0, k : int = 10) : 
    step_size = math.floor(shuffled_data.shape[0]/k)
    start_idx = i*step_size
    end_idx = i*step_size+step_size
    testset = shuffled_data[start_idx : end_idx-1]
    trainset = shuffled_data.drop(range(start_idx, end_idx), axis=0)
    return trainset, testset