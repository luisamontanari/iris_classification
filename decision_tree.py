import pandas as pd
import math

class decision_tree:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data
    
    def traverse(self, datapoint):
        #traverse tree until we get to a leaf, then output classification label
        if isinstance(self.data, leaf) :
            return self.data.label
        
        node = self.data
        if datapoint[node.feature] < node.threshold :
            return self.left.traverse(datapoint)
        
        return self.right.traverse(datapoint)
    
    # TODO: fix show task, currently only breadth-first traversal
    def show_tree(self) :
        if isinstance(self.data, tree_node) :
            data = ['TreeNode:', self.data.threshold, self.data.feature]
            if (self.left and self.right) :
                #print('both children')
                return f'{data} --- {self.left.show_tree()} --- {self.right.show_tree()}'
            if (self.left) :
                return f'{data}\n{self.left.show_tree()} --- Error on right child'
            if (self.right) :
                return f'{data}\n Error on left child --- {self.right.show_tree()}'
            return 'Error on both children'
        if isinstance(self.data, leaf) :
            data = ['Leaf: ', self.data.label]
            return data

class leaf:
    def __init__(self, label):
        self.label = label

class tree_node:
    def __init__(self, threshold, feature,):
        self.threshold = threshold
        self.feature = feature


def calc_gini(df, avg, variety, feature) :
    # step 2: separate df by threshold
    left_branch = df[df[feature] < avg]
    right_branch = df[df[feature] >= avg]
    
    #print(left_branch.shape[0], right_branch.shape[0])
    if left_branch.shape[0] == 0 or right_branch.shape[0] == 0: return 1, True

    # step 3: separate branches by variety (yes/no)
    l_vc_yes = left_branch[left_branch['variety'] == variety].shape[0]
    r_vc_yes = right_branch[right_branch['variety'] == variety].shape[0]

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

def calc_min_gini_impurity(df, feature) : 
    min_gini = [1,1,1]
    threshold = 0
    direction = False
    gini= [1,1,1] #todo: this should always correspond to the unique numbrr  of features, make this nicer
    dir = [True, True, True]
    for i in range(df.shape[0]-1) :
        # step 1: calculate average between pairs in the sorted feature column as potential thresholds
        avg = (df[feature].iloc[i] + df[feature].iloc[i+1])/2
    
        for i, variety in enumerate(df['variety'].unique()): ## TODO:vectorize?
            gini[i], dir[i] = calc_gini(df, avg, variety, feature)

        if min(gini) < min(min_gini) : 
            min_gini = gini.copy()
            threshold = avg
            direction =  dir.copy()
            
    
    return(min_gini, threshold, direction)

def get_purest_node(df, verbose=False) :
    min_gini_total = [1,1,1]
    direction = True
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
            direction = dir
            purest_threshold = threshold
    if (verbose) : print('-----------------------------------')
    return min_gini_total, purest_threshold, feat, direction

def build_decision_tree(trainset, g_i=1, isUnderThreshold=False, variety=None, verbose=False) :
    #TODO: add speciaal case: if only one feature exists, return leaf node immediately

    g_i, threshold, feature, dir = get_purest_node(trainset, verbose=verbose)

    node = decision_tree(tree_node(threshold, feature))

    # recursively calc purest node on both child segmentions: 
    trainset_left = trainset[trainset[feature] < threshold]
    trainset_right = trainset[trainset[feature] >= threshold]
    
    for i in range(3) :
        if g_i[i] < 0.15: # TODO: add Rekursionanker with a sensible condition (if GI is low or the number of people is low or the tree is too big  etc)
            if dir[i] : 
                variety = trainset_left['variety'].value_counts().idxmax()
                node.left = decision_tree(leaf(variety))
            else : 
                variety = trainset_right['variety'].value_counts().idxmax()
                node.right = decision_tree(leaf(variety))
    if (not node.right):
        node.right = build_decision_tree(trainset_right, g_i, isUnderThreshold, variety, verbose=verbose)
    if (not node.left) :
        node.left = build_decision_tree(trainset_left, g_i, isUnderThreshold, variety, verbose=verbose)
    return node
