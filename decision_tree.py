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

def calc_gini_impurity(df, feature, variety) : 
    min_gini = 1
    w_g_l = 1
    w_g_r = 1
    threshold = 0
    direction = False
    for i in range(df.shape[0]-1) :
        # step 1: calculate average between pairs in the sorted feature column as potential thresholds
        avg = (df[feature].iloc[i] + df[feature].iloc[i+1])/2
    
        # step 2: separate df by threshold
        left_branch = df[df[feature] < avg]
        right_branch = df[df[feature] >= avg]
        
        #print(left_branch.shape[0], right_branch.shape[0])
        if left_branch.shape[0] == 0 or right_branch.shape[0] == 0: continue
    
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
        if total_gini < min_gini : 
            min_gini = total_gini
            w_g_l = w_gini_left
            w_g_r = w_gini_right
            threshold = avg
            direction =  l_vc_yes > r_vc_yes
    
    return(min_gini, w_g_l, w_g_r, threshold, direction)

def get_purest_node(df, verbose=False) :
    min_gini_total = (1, math.inf, math.inf, math.nan, '')
    feat = ''
    iris_var = ''
    for variety in df['variety'].unique() : 
        if (verbose) : print(f'\n{variety}:')
        for feature in df.columns[:-1]:
            sorted_df = df.sort_values(by=[feature], ignore_index=True)
            g_i = calc_gini_impurity(sorted_df, feature, variety)
            if (verbose) : print(f'Gini for {feature}: {g_i}')
            if g_i[0] < min_gini_total[0] :
                min_gini_total = g_i
                feat = feature
                iris_var = variety
    if min_gini_total == 1 : 
        print(f"ERROR: Gini = 1")
        exit(1)
    return *min_gini_total, feat, iris_var

def build_decision_tree(trainset, g_i=1, isUnderThreshold=False, variety=None, verbose=False) :

    g_i, g_left, g_right, threshold, isUnderThreshold, feature, variety = get_purest_node(trainset, verbose=verbose)
    print(g_i, g_left, g_right, threshold, isUnderThreshold, feature, variety)

    if g_i == 1 : 
        node = decision_tree(leaf(variety))
        return node

#    tree_node 
#L:Set  --   tree_node
#        tree_node --- L:Virg
#    L:Vers -- tn
#          l:Virg -- L:ERROR
    
    node = decision_tree(tree_node(threshold, feature))

    # recursively calc purest node on both child segmentions: 
    trainset_left = trainset[trainset[feature] < threshold]
    trainset_right = trainset[trainset[feature] >= threshold]
    
    ## PROBLEM: What happens if a feature neatly separates two varieties??
    
    # TODO: add Rekursionanker with a sensible condition (if GI is low or the number of people is low or the tree is too big  etc)
    
    if g_i < 0.15:
        # TODO I don't think we actually need to remember the variety, we can just take the variety with the most data point in the result
        # maybe we don't need the direction either?  Think this thorugh

        ###if g_left < 0.1 : 
        ###    node.left = decision_tree(leaf(variety))
        ###else : node.left = build_decision_tree(trainset_left, g_i, isUnderThreshold, variety, verbose=verbose)
        ###if g_right < 0.1 : 
        ###    node.right = decision_tree(leaf(variety))
        ###else : node.right = build_decision_tree(trainset_right, g_i, isUnderThreshold, variety, verbose=verbose)
            
        if isUnderThreshold : 
            node.left = decision_tree(leaf(variety))
            node.right = build_decision_tree(trainset_right, g_i, isUnderThreshold, variety, verbose=verbose)
        else : 
            node.right = decision_tree(leaf(variety))
            node.left = build_decision_tree(trainset_left, g_i, isUnderThreshold, variety, verbose=verbose)
        #node = decision_tree(leaf(variety))
        # problem::
        return node
    #else :
    node.left = build_decision_tree(trainset_left, g_i, isUnderThreshold, variety, verbose=verbose)
    node.right = build_decision_tree(trainset_right, g_i, isUnderThreshold, variety, verbose=verbose)

    return node


