import numpy as np
from sklearn.metrics import cohen_kappa_score
import pandas as pd

def compute_cohen_kappa(annotator1: list, annotator2: list) -> float:
    """
    Computes Cohen's Kappa for two annotators.
    Works for both categorical (Intent) and ordinal (Correctness) labels.
    """
    return cohen_kappa_score(annotator1, annotator2)

def compute_fleiss_kappa(ratings: np.ndarray, n_categories: int) -> float:
    """
    Computes Fleiss' Kappa for multiple annotators.
    ratings: 2D array (items x annotators) containing category indices.
    """
    n_items, n_annotators = ratings.shape
    
    # Create the item-category matrix
    mat = np.zeros((n_items, n_categories))
    for i in range(n_items):
        for j in range(n_annotators):
            mat[i, ratings[i, j]] += 1
            
    # Calculate p_j (proportion of all assignments which were to the j-th category)
    p_j = mat.sum(axis=0) / (n_items * n_annotators)
    
    # Calculate P_i (extent to which annotators agree for the i-th subject)
    P_i = (np.sum(mat * mat, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    
    P_bar = P_i.mean()
    P_e_bar = np.sum(p_j * p_j)
    
    if P_e_bar == 1:
        return 1.0
        
    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
    return kappa

def adjudicate_annotations(annotations_df: pd.DataFrame, key_column: str, label_column: str) -> pd.DataFrame:
    """
    Simple majority voting adjudication. 
    If there is a tie, it marks it as 'NEEDS_REVIEW'.
    """
    def majority_vote(group):
        counts = group[label_column].value_counts()
        if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
            return 'NEEDS_REVIEW'
        return counts.index[0]
        
    return annotations_df.groupby(key_column).apply(majority_vote).reset_index(name='adjudicated_' + label_column)
