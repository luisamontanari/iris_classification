import pandas as pd
import math
import numpy as np
from models.ml_model import *

# TODO research classification and regression tree algorithm (CART)

class decision_tree(ml_model):
    def __init__(self, data, verbose=False):
        self.root = self._build_decision_tree(data, verbose)
        
    def __str__(self) : 
        return self.root._show_tree()
    
    # classify a single datapoint
    def classify_datapoint(self, datapoint : pd.Series) -> str :
        return self.root._classify_datapoint(datapoint)
    
    def classify_dataset(self, df : pd.core.frame.DataFrame) -> pd.core.frame.DataFrame :
        return self.root._classify_dataset(df)
    
    # calc GI on a specific segementation of the dataset (left_branch vs right_branch)
    def _calc_GI(self, df, left_branch, right_branch, unique_feat_count) :
        l_variety_occ_arr = np.arange(unique_feat_count) # will contain occurence count of each variety in left_branch
        r_variety_occ_arr = np.arange(unique_feat_count)

        for i, variety in enumerate(df['variety'].unique()):
            l_variety_occ_arr[i] = left_branch[left_branch['variety'] == variety].shape[0]
            r_variety_occ_arr[i] = right_branch[right_branch['variety'] == variety].shape[0]

        l_variety_miss_arr = left_branch.shape[0] - l_variety_occ_arr
        r_variety_miss_arr = right_branch.shape[0] - r_variety_occ_arr

        # step 5 calculate unweighted Gini Impurity
        gini_left = 1 - (l_variety_occ_arr/left_branch.shape[0])**2 - (l_variety_miss_arr/left_branch.shape[0])**2
        gini_right = 1 - (r_variety_occ_arr/right_branch.shape[0])**2 - (r_variety_miss_arr/right_branch.shape[0])**2

        # weighted left and right gini impurities
        w_gini_left = (left_branch.shape[0]/df.shape[0]) * gini_left
        w_gini_right = (right_branch.shape[0]/df.shape[0]) * gini_right

        total_gini = w_gini_left + w_gini_right
        return total_gini, l_variety_occ_arr > r_variety_occ_arr

    # calculate the threshold with minimal GI on a specific feature
    def _calc_min_GI_for_feature(self, df : pd.core.frame.DataFrame, feature : str) -> tuple[list[int], int, bool]: 
        unique_feat_count = df['variety'].unique().shape[0]
        min_gini = [1] * unique_feat_count
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
            
            gini, dir = self._calc_GI(df, left_branch, right_branch, unique_feat_count)

            if min(gini) < min(min_gini) : 
                min_gini = gini.copy()
                threshold = avg
                direction = dir.copy()

        return min_gini, threshold, direction

    # calculate feature and threshold with minimal GI
    def _calc_purest_segmentation(self, df, verbose=False) :
        unique_feat_count = df['variety'].unique().shape[0]
        min_gini_total = [1] * unique_feat_count
        for feature in df.columns[:-1]:
            if (verbose) : print(f'\n{feature}:')
            sorted_df = df.sort_values(by=[feature], ignore_index=True)
            g_i, threshold, dir = self._calc_min_GI_for_feature(sorted_df, feature)
            if (verbose) : print(f'Gini coefficients: {g_i}, direction: {dir}')
            if min(g_i) < min(min_gini_total) :
                min_gini_total = g_i.copy()
                purest_feature = feature
                direction = dir.copy()
                purest_threshold = threshold
        if (verbose) : print('-----------------------------------')
        return min_gini_total, purest_threshold, purest_feature, direction

    # if df contains only one variety, return that. Otherwise, return None
    def _get_unique_variety_if_ex(self, df):
        df_unique = df['variety'].unique()
        if len(df_unique) == 1: 
            return df_unique['variety'].iloc[0]
        return None

    def _build_decision_tree(self, trainset, g_i=[], variety=None, verbose=False) :

        #special case: if only one feature exists, return leaf node immediately
        unique_variety = self._get_unique_variety_if_ex(trainset)
        if unique_variety :  
            node = leaf(unique_variety)
            return node

        # determine purest threshold and g_i value for each variety when separated by that threshold
        g_i, threshold, feature, most_entries_under_threshold = self._calc_purest_segmentation(trainset, verbose=verbose)

        node = decision_node(threshold, feature) 

        # split trainset by threshold
        trainset_left = trainset[trainset[feature] < threshold]
        trainset_right = trainset[trainset[feature] >= threshold]

        # for each variety, check if respective g_i is pure enough to create leaf node
        for i in range(trainset['variety'].unique().shape[0]) :
            if g_i[i] < 0.15: # TODO: add second condition to avoid model blowup (like tree depth)
                if most_entries_under_threshold[i] : 
                    variety = trainset_left['variety'].value_counts().idxmax() # the most common variety in data segment becomes the leaf label
                    node.left = leaf(variety)
                else : 
                    variety = trainset_right['variety'].value_counts().idxmax()
                    node.right = leaf(variety)
        # if left/right node is not a leaf, recurse -> it will then automatically become a decision node
        if (not node.right):
            node.right = self._build_decision_tree(trainset_right, g_i, variety, verbose=verbose)
        if (not node.left) :
            node.left = self._build_decision_tree(trainset_left, g_i, variety, verbose=verbose)
        return node

class tree_node: 
     # classify a single datapoint
    def _classify_datapoint(self, datapoint):
        # TODO: add exception if child node does not exist
        #traverse tree until we get to a leaf, then output classification label
        if isinstance(self, leaf) :
            return self.label
        
        # we can now be sure that self is a decision_node
        if datapoint[self.feature] < self.threshold :
            return self.left._classify_datapoint(datapoint)
        
        return self.right._classify_datapoint(datapoint)

    def _classify_dataset(self, df : pd.core.frame.DataFrame) -> pd.core.frame.DataFrame :
        series = df.apply(self._classify_datapoint, axis=1)
        return pd.DataFrame(data=series, columns=['prediction'])
    
    # show tree in breadth-first-traversal
    def _show_tree(self) :
        if isinstance(self, decision_node) :
            data = f'(TreeNode: {self.feature} {self.threshold:.3f})'
            if (self.left and self.right) :
                return f'{data} --- {self.left._show_tree()} --- {self.right._show_tree()}'
            if (self.left) :
                return f'{data}\n{self.left._show_tree()} --- Error on right child'
            if (self.right) :
                return f'{data}\n Error on left child --- {self.right._show_tree()}'
            return 'Error on both children'
        if isinstance(self, leaf) :
            data = f'(Leaf: {self.label})'
            return data

class leaf(tree_node):
    def __init__(self, label):
        self.label = label

class decision_node(tree_node):
    def __init__(self, threshold, feature):
        self.left = None
        self.right = None
        self.threshold = threshold
        self.feature = feature