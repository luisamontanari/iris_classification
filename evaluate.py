from decision_tree import *

def evaluate_model(model, dataset) :
    dataset = model.classify_dataset(dataset)
    df = dataset.apply(lambda dp : int(dp['variety'] == dp['classification']), axis=1)
    accuracy = df.sum()/df.count()
    return accuracy
