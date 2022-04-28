import numpy as np
from typing import Optional

def get_subset_indices(x, n_subset: int=10, seed: Optional[int]=123):
    rng = np.random.RandomState(seed)
    ind = rng.choice(np.arange(x.shape[0]), size=(n_subset))
    return ind
    