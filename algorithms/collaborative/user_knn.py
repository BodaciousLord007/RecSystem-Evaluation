import numpy as np
from ..base import BaseRecommender


def overlap_similarity(matrix, shrinkage=10):
    """Pearson on co-rated entries; absent ratings are not zero observations."""
    mask = matrix.copy()
    mask.data = np.ones_like(mask.data)
    squared = matrix.copy()
    squared.data **= 2
    count = (mask @ mask.T).toarray()
    sums = (matrix @ mask.T).toarray()
    sums2 = (squared @ mask.T).toarray()
    products = (matrix @ matrix.T).toarray()
    safe = np.maximum(count, 1)
    cov = products - sums * sums.T / safe
    variance = np.maximum(sums2 - sums ** 2 / safe, 0)
    denom = np.sqrt(variance * variance.T)
    sim = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 1e-10)
    # A near-perfect match on very few shared books is weak evidence.
    sim *= count / (count + shrinkage)
    sim[count < 3] = 0
    np.fill_diagonal(sim, 0)
    return sim


class UserKNN(BaseRecommender):
    rating_model = True

    def __init__(self, k=30):
        self.k = k

    def fit(self, ratings, n_users, n_items, features=None):
        super().fit(ratings, n_users, n_items, features)
        self.sim = overlap_similarity(self.matrix)
        self.columns = self.matrix.tocsc()  # Fast lookup of each book's raters.
        return self

    def score(self, user, items):
        predictions = []
        for item in items:
            start, end = self.columns.indptr[item:item + 2]
            peers = self.columns.indices[start:end]
            values = self.columns.data[start:end]
            weights = self.sim[user, peers]
            # Use the strongest positive neighbours who actually rated this book.
            selected = np.flatnonzero(weights > 0)
            selected = selected[np.argsort(weights[selected])[-self.k:]]
            # print('Target user/item and usable neighbours:', user, item, len(selected))
            if len(selected):
                # Account for neighbours who consistently rate high or low.
                residual = values[selected] - self.user_mean[peers[selected]]
                estimate = self.user_mean[user] + np.dot(weights[selected], residual) / weights[selected].sum()
            else:
                estimate = self.user_mean[user]
            predictions.append(np.clip(estimate, 1, 5))
        return np.array(predictions)

