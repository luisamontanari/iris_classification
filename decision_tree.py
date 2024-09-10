import pandas as pd

class decision_tree:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

class leaf:
    def __init__(self, label):
        self.label = label

class tree_node:
    def __init__(self, threshold, feature,):
        self.threshold = threshold
        self.feature = feature

def calc_gini_impurity(df, feature, variety) : 
    min_gini = 1
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
    
        # step 5 calculate Gini Impurity
        gini_left = 1 - (l_vc_yes/left_branch.shape[0])**2 - (l_vc_no/left_branch.shape[0])**2
        gini_right = 1 - (r_vc_yes/right_branch.shape[0])**2 - (r_vc_no/right_branch.shape[0])**2
    
        total_gini = (left_branch.shape[0]/df.shape[0]) * gini_left + (right_branch.shape[0]/df.shape[0]) * gini_right
        if total_gini < min_gini : 
            min_gini = total_gini
            threshold = avg
            direction =  l_vc_yes > r_vc_yes
    
    return(min_gini, threshold, direction)

def get_purest_node(df, verbose=False) :
    min_gini_total = (1, 1, '')
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
    return *min_gini_total, feat, iris_var



