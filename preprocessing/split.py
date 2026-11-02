import numpy as np
import pandas as pd


def split_ratings(ratings, seed=42):
    # Goodbooks has no rating timestamps, so split randomly within each user.
    rng = np.random.default_rng(seed)
    parts = [[], [], []]  # Train, validation, test.
    for _, group in ratings.groupby('user', sort=True):
        order = rng.permutation(len(group))
        n_test = max(1, int(len(group) * .15))
        n_val = max(1, int(len(group) * .15))
        if len(group) - n_test - n_val < 2:
            raise ValueError('Each user needs at least four observations.')
        parts[0].append(group.iloc[order[n_test + n_val:]])
        parts[1].append(group.iloc[order[n_test:n_test + n_val]])
        parts[2].append(group.iloc[order[:n_test]])
    return tuple(pd.concat(part, ignore_index=True) for part in parts)

