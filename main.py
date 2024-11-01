import sys
import time

from models.decision_tree import *
from models.naive_bayes import *
from models.kNN import *
from helper import *


def main() : 
    pd.options.mode.chained_assignment = None # disable SettingWithCopyWarning
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)}) # format np array prints to 2 decimal places
    verbose = False
    debug = False
    if 'v' in sys.argv: verbose = True
    if 'd' in sys.argv: debug = True

    iris_data = iris_data = pd.read_csv('./input/iris.csv')
    shuffled_data = iris_data.sample(frac=1).reset_index(drop=True) # shuffle data

    k=10 # param for k-fold-cross-validation, commonly 10
    model_mapping = {0: 'Decision Tree', 1: 'Naive Bayes', 2: 'k-nearest neighbor'}
    accuracy = np.zeros((k, len(model_mapping)))

    start = time.time() # TODO: function specific timer would be neat
    #-----------------------------------------------------------
    # k-fold cross-validation
    for i in range(k) : 
        trainset, testset = get_train_test_split(shuffled_data, i, k)

        ##--------------- decision tree ----------------------
        root = decision_tree(trainset, debug=debug)
        accuracy[i, 0] = evaluate_model(root, testset)
        if verbose : print(f'Decision tree representation in breadth-first traversal for iteration {i}: {root}')

        ##--------------- naive bayes ----------------------
        bayes_model = naive_bayes_model(shuffled_data)
        accuracy[i, 1] = evaluate_model(bayes_model, testset)

        ##--------------- k-nearest neighbor ----------------------
        kNN = kNN_model(shuffled_data, k=3)
        accuracy[i, 2] = evaluate_model(kNN, testset)

        if verbose : print(f'Accuracy for iteration {i}: {accuracy[i]}')

    avg_model_accuracy = np.average(accuracy, axis=0)
    end = time.time()

    for i in range(len(model_mapping)) :
        print(f'Average model accuracy for {model_mapping.get(i)}: {avg_model_accuracy[i]:.2f}')
    print(f'Total execution time for building and evaluating all models: {end - start:.3f} seconds')
    
if __name__ == '__main__':
    main()



