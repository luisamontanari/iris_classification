from models.decision_tree import *

def evaluate_model(model, dataset) :
    prediction = model.classify_dataset(dataset)
    merged_df = pd.concat([dataset, prediction], axis=1)
    correct_classifications = merged_df.apply(lambda dp : int(dp['variety'] == dp['prediction']), axis=1)
    accuracy = correct_classifications.sum()/dataset.shape[0]
    return accuracy
