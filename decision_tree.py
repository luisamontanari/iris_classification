import pandas as pd
import math
import numpy as np

# TODO research classification and regression tree algorithm (CART)

# TODO: implement ml_model parent class
# utilize constructor instead of outside function build_decision_tree / or reference build_decision_tree in constr
class decision_tree:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data
        
    def __str__(self) : 
        return self.__show_tree()
    
    # classify a single datapoint
    def classify_datapoint(self, datapoint):
        # TODO: add exception if child node does not exist
        #traverse tree until we get to a leaf, then output classification label
        if isinstance(self.data, leaf) :
            return self.data.label
        
        node = self.data
        if datapoint[node.feature] < node.threshold :
            return self.left.classify_datapoint(datapoint)
        
        return self.right.classify_datapoint(datapoint)
    
    def classify_dataset(self, df : pd.core.frame.DataFrame) -> pd.core.frame.DataFrame :
        series = df.apply(self.classify_datapoint, axis=1)
        return pd.DataFrame(data=series, columns=['prediction'])
    
    # TODO: fix show task, currently only breadth-first traversal
    def __show_tree(self) :
        if isinstance(self.data, tree_node) :
            data = f'(TreeNode: {self.data.feature} {self.data.threshold:.3f})'
            if (self.left and self.right) :
                return f'{data} --- {self.left.__show_tree()} --- {self.right.__show_tree()}'
            if (self.left) :
                return f'{data}\n{self.left.__show_tree()} --- Error on right child'
            if (self.right) :
                return f'{data}\n Error on left child --- {self.right.__show_tree()}'
            return 'Error on both children'
        if isinstance(self.data, leaf) :
            data = f'(Leaf: {self.data.label})'
            return data

class leaf:
    def __init__(self, label):
        self.label = label

class tree_node:
    def __init__(self, threshold, feature,):
        self.threshold = threshold
        self.feature = feature

def calc_gini_vectorized(df, left_branch, right_branch, len) :
    l_vc_yes = np.arange(len)
    r_vc_yes = np.arange(len)
    
    for i, variety in enumerate(df['variety'].unique()):
        l_vc_yes[i] = left_branch[left_branch['variety'] == variety].shape[0]
        r_vc_yes[i] = right_branch[right_branch['variety'] == variety].shape[0]

    l_vc_no = left_branch.shape[0] - l_vc_yes
    r_vc_no = right_branch.shape[0] - r_vc_yes

    # step 5 calculate unweighted Gini Impurity
    gini_left = 1 - (l_vc_yes/left_branch.shape[0])**2 - (l_vc_no/left_branch.shape[0])**2
    gini_right = 1 - (r_vc_yes/right_branch.shape[0])**2 - (r_vc_no/right_branch.shape[0])**2
    
    # weighted left and right gini impurities
    w_gini_left = (left_branch.shape[0]/df.shape[0]) * gini_left
    w_gini_right = (right_branch.shape[0]/df.shape[0]) * gini_right

    total_gini = w_gini_left + w_gini_right
    return total_gini, l_vc_yes > r_vc_yes

def calc_min_gini_impurity(df, feature) : # TODO: vectorize
    len = df['variety'].unique().shape[0]
    min_gini = [1] * len
    threshold = 0
    direction = [True] * len
    gini= [1] * len
    dir = [True] * len
    prev_avg = math.nan
    for i in range(df.shape[0]-1) :
        # step 1: calculate average between pairs in the sorted feature column as potential thresholds
        avg = (df[feature].iloc[i] + df[feature].iloc[i+1])/2
        if avg == prev_avg: 
            continue
        prev_avg = avg
        
        # step 2: separate df by threshold
        left_branch = df[df[feature] < avg]
        right_branch = df[df[feature] >= avg]
        
        if left_branch.shape[0] == 0 or right_branch.shape[0] == 0: 
            continue
        
        gini, dir = calc_gini_vectorized(df, left_branch, right_branch, len)

        if min(gini) < min(min_gini) : 
            min_gini = gini.copy()
            threshold = avg
            direction = dir.copy()
            
    return(min_gini, threshold, direction)

def get_purest_node(df, verbose=False) :
    len = df['variety'].unique().shape[0]
    min_gini_total = [1] * len
    direction = [True] * len
    purest_threshold = 0
    feat = ''
    for feature in df.columns[:-1]:
        if (verbose) : print(f'\n{feature}:')
        sorted_df = df.sort_values(by=[feature], ignore_index=True)
        g_i, threshold, dir = calc_min_gini_impurity(sorted_df, feature)
        if (verbose) : print(f'Gini coefficients: {g_i}, direction: {dir}')
        if min(g_i) < min(min_gini_total) :
            min_gini_total = g_i.copy()
            feat = feature
            direction = dir.copy()
            purest_threshold = threshold
    if (verbose) : print('-----------------------------------')
    return min_gini_total, purest_threshold, feat, direction

# if df contains only one variety, return that. Otherwise, return None
def get_variety_if_all_eq(df):
    df_unique = df['variety'].unique()
    if len(df_unique) == 1: 
        return df_unique['variety'].iloc[0]
    return None

def build_decision_tree(trainset, g_i=1, variety=None, verbose=False) :
    
    #special case: if only one feature exists, return leaf node immediately
    unique_variety = get_variety_if_all_eq(trainset)
    if unique_variety :  
        node = decision_tree(leaf(unique_variety))
        return node

    # determine purest threshold and g_i value for each variety when separated by that threshold
    g_i, threshold, feature, most_entries_under_threshold = get_purest_node(trainset, verbose=verbose)

    node = decision_tree(tree_node(threshold, feature))

    # split trainset by threshold
    trainset_left = trainset[trainset[feature] < threshold]
    trainset_right = trainset[trainset[feature] >= threshold]
    
    # for each variety, check if respective g_i is pure enough to create leaf node
    for i in range(trainset['variety'].unique().shape[0]) :
        if g_i[i] < 0.15: # TODO: add Rekursionanker with a sensible condition (if GI is low or the number of people is low or the tree is too big  etc)
            if most_entries_under_threshold[i] : 
                variety = trainset_left['variety'].value_counts().idxmax()
                node.left = decision_tree(leaf(variety))
            else : 
                variety = trainset_right['variety'].value_counts().idxmax()
                node.right = decision_tree(leaf(variety))
    # if left/right node is not a leaf, recurse -> it will then automatically become a decision node
    if (not node.right):
        node.right = build_decision_tree(trainset_right, g_i, variety, verbose=verbose)
    if (not node.left) :
        node.left = build_decision_tree(trainset_left, g_i, variety, verbose=verbose)
    return node
