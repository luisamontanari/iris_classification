import pandas as pd
import math
import numpy as np
from models.ml_model import *
from typing import Optional

# Node in the decision tree. Can be either a leaf node containing a label, or a decision_node containing a threshold and two children
class tree_node: 
     # classify a single datapoint
    def _classify_datapoint(self, datapoint : pd.Series) -> str :
        #traverse tree until we get to a leaf, then output classification label
        if isinstance(self, leaf) :
            return self.label
        
        # we can now be sure that self is a decision_node
        if datapoint[self.feature] < self.threshold :
            return self.left._classify_datapoint(datapoint)
        
        return self.right._classify_datapoint(datapoint)

    def _classify_dataset(self, df : pd.DataFrame) -> pd.DataFrame :
        series = df.apply(self._classify_datapoint, axis=1)
        return pd.DataFrame(data=series, columns=['prediction'])

    # show tree in breadth-first-traversal
    def _show_tree(self) -> str :
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

class decision_tree(ml_model):
    def __init__(self, training_data, debug=False):
        self.root : tree_node = self._build_decision_tree(training_data, debug=debug)
        
    def __str__(self) : 
        return self.root._show_tree()
    
    # classify a single datapoint
    def classify_datapoint(self, datapoint : pd.Series) -> str :
        return self.root._classify_datapoint(datapoint)
    
    def classify_dataset(self, df : pd.DataFrame) -> pd.DataFrame :
        return self.root._classify_dataset(df)
    
    # calc GI on a specific segementation of the dataset (left_branch vs right_branch)
    def _calc_GI(self, left_branch, right_branch, unique_feat_list) ->  tuple[list[int], bool]:
        # fill arrays with occurence count of each variety in left/right_branch
        unique_feat_count = unique_feat_list.shape[0]
        l_variety_occ_arr = np.arange(unique_feat_count)
        r_variety_occ_arr = np.arange(unique_feat_count)
        
        #for i, variety in enumerate(df['variety'].unique()):
        for i, variety in enumerate(unique_feat_list):
            l_variety_occ_arr[i] = left_branch[left_branch['variety'] == variety].shape[0]
            r_variety_occ_arr[i] = right_branch[right_branch['variety'] == variety].shape[0]

        l_variety_miss_arr = left_branch.shape[0] - l_variety_occ_arr
        r_variety_miss_arr = right_branch.shape[0] - r_variety_occ_arr

        # calculate unweighted Gini Impurity
        gini_left = 1 - (l_variety_occ_arr/left_branch.shape[0])**2 - (l_variety_miss_arr/left_branch.shape[0])**2
        gini_right = 1 - (r_variety_occ_arr/right_branch.shape[0])**2 - (r_variety_miss_arr/right_branch.shape[0])**2

        # calc weighted left and right gini impurities
        n = left_branch.shape[0] + right_branch.shape[0]
        w_gini_left = (left_branch.shape[0]/n) * gini_left
        w_gini_right = (right_branch.shape[0]/n) * gini_right

        total_gini = w_gini_left + w_gini_right
        return total_gini, l_variety_occ_arr > r_variety_occ_arr

    # calculate the threshold with minimal GI on a specific feature
    def _calc_min_GI_for_feature(self, df : pd.DataFrame, feature : str) -> tuple[list[int], int, bool]: 
        unique_feat_list = df['variety'].unique()
        min_gini = [1] * unique_feat_list.shape[0]
        threshold = None
        direction = None
        prev_avg = math.nan
        for i in range(df.shape[0]-1) :
            # calculate average between pairs in the sorted feature column as potential thresholds
            avg = (df[feature].iloc[i] + df[feature].iloc[i+1])/2
            if avg == prev_avg: 
                continue
            prev_avg = avg

            # separate df by threshold
            left_branch = df[df[feature] < avg]
            right_branch = df[df[feature] >= avg]

            if left_branch.shape[0] == 0 or right_branch.shape[0] == 0: 
                continue
            
            # calc GI for separated data
            gini, dir = self._calc_GI(left_branch, right_branch, unique_feat_list)

            if min(gini) <= min(min_gini) : 
                min_gini = gini.copy()
                threshold = avg
                direction = dir.copy()

        return min_gini, threshold, direction

    # calculate feature and threshold with minimal GI
    def _calc_purest_segmentation(self, df : pd.DataFrame, debug : bool = False) -> tuple[list[int], int, str, bool]:
        unique_feat_count = df['variety'].unique().shape[0]
        min_gini_total = [1] * unique_feat_count
        for feature in df.columns[:-1]:
            if (debug) : print(f'\n{feature}:')
            sorted_df = df.sort_values(by=[feature], ignore_index=True)
            g_i, threshold, dir = self._calc_min_GI_for_feature(sorted_df, feature)
            if (debug) : print(f'Gini coefficients: {g_i}, direction: {dir}')
            if min(g_i) < min(min_gini_total) :
                min_gini_total = g_i.copy()
                purest_feature = feature
                direction = dir.copy()
                purest_threshold = threshold
        if (debug) : print('-----------------------------------')
        return min_gini_total, purest_threshold, purest_feature, direction

    # if df contains only one variety, return that. Otherwise, return None
    def _get_unique_variety_if_ex(self, df : pd.DataFrame) -> Optional[str]:
        df_unique = df['variety'].unique()
        if len(df_unique) == 1: return df_unique[0]
        return None

    # build decision tree model
    def _build_decision_tree(self, trainset : pd.DataFrame, recursion_depth : int = 0, variety : str = None, debug : bool = False) -> tree_node :
        #special case: if only one feature exists, return leaf node immediately
        unique_variety = self._get_unique_variety_if_ex(trainset)
        if unique_variety : return leaf(unique_variety)

        # determine purest threshold and g_i value for each variety when separated by that threshold
        g_i, threshold, feature, most_entries_under_threshold = self._calc_purest_segmentation(trainset, debug=debug)

        node = decision_node(threshold, feature) 

        # split trainset by threshold
        trainset_left = trainset[trainset[feature] < threshold]
        trainset_right = trainset[trainset[feature] >= threshold]

        # for each variety, check if respective g_i is pure enough to create leaf node
        for i in range(trainset['variety'].unique().shape[0]) :
            if g_i[i] < 0.15 or recursion_depth >= 1:
                if most_entries_under_threshold[i] : 
                    variety = trainset_left['variety'].value_counts().idxmax() # the most common variety in data segment becomes the leaf label
                    node.left = leaf(variety)
                else : 
                    variety = trainset_right['variety'].value_counts().idxmax()
                    node.right = leaf(variety)
        # if left/right node is not a leaf, recurse -> it will then automatically become a decision node
        if (not node.right):
            node.right = self._build_decision_tree(trainset_right, recursion_depth+1, variety, debug)
        if (not node.left) :
            node.left = self._build_decision_tree(trainset_left, recursion_depth+1, variety, debug)
        return node
