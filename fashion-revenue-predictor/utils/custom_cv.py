import numpy as np
from sklearn.model_selection import BaseCrossValidator
from typing import List, Tuple, Optional

class GroupTimeSeriesSplit(BaseCrossValidator):
    """
    Custom cross-validation class that implements grouped time series splitting.
    This ensures that data from the same store/group stays together in train/test splits.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits for cross-validation
    test_size : float, default=0.2
        Proportion of data to use for testing in each split
    """
    
    def __init__(self, n_splits: int = 5, test_size: float = 0.2):
        self.n_splits = n_splits
        self.test_size = test_size
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splits for cross-validation."""
        return self.n_splits
    
    def split(self, X, y=None, groups=None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test sets.
        
        Parameters
        ----------
        X : array-like
            Training data
        y : array-like, optional
            Target values
        groups : array-like
            Group labels for the samples (e.g., store IDs)
            
        Yields
        ------
        train : ndarray
            Training set indices for that split
        test : ndarray
            Test set indices for that split
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None")
        
        # Get unique groups and their sizes
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        # Calculate number of groups for test set
        n_test_groups = max(1, int(n_groups * self.test_size))
        
        # Generate splits
        for i in range(self.n_splits):
            # Calculate start and end indices for test set
            test_start = i * n_test_groups
            test_end = min((i + 1) * n_test_groups, n_groups)
            
            # Get test groups
            test_groups = unique_groups[test_start:test_end]
            
            # Create boolean masks for train and test
            test_mask = np.isin(groups, test_groups)
            train_mask = ~test_mask
            
            # Get indices
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            yield train_idx, test_idx 