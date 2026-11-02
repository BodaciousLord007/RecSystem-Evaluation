import numpy as np
from ..base import BaseRecommender


class ItemKNN(BaseRecommender):
    rating_model = True

    def __init__(self, k=30):
        self.k = k

    def fit(self, ratings, n_users, n_items, features=None):
        super().fit(ratings, n_users, n_items, features)
        # Remove each user's rating bias before comparing books.
        centered = self.matrix.copy()
        centered.data -= np.repeat(self.user_mean, np.diff(centered.indptr))
        mask = self.matrix.copy()
        mask.data = np.ones_like(mask.data)
        squared = centered.copy()
        squared.data **= 2
        products = (centered.T @ centered).toarray()
        # Norms use only users who rated both books.
        sums2 = (squared.T @ mask).toarray()
        overlap = (mask.T @ mask).toarray()
        denom = np.sqrt(np.maximum(sums2 * sums2.T, 0))
        self.sim = np.divide(products, denom, out=np.zeros_like(products), where=denom > 1e-10)
        # Downweight pairs with little shared history.
        self.sim *= overlap / (overlap + 10)
        self.sim[overlap < 3] = 0
        np.fill_diagonal(self.sim, 0)
        return self

    def score(self, user, items):
        history = self.matrix.getrow(user)
        predictions = []
        for item in items:
            # Neighbours must come from this user's rated books.
            weights = self.sim[item, history.indices]
            selected = np.flatnonzero(weights > 0)
            selected = selected[np.argsort(weights[selected])[-self.k:]]
            if len(selected):
                residual = history.data[selected] - self.user_mean[user]
                estimate = self.user_mean[user] + np.dot(weights[selected], residual) / weights[selected].sum()
            else:
                estimate = self.user_mean[user]
            predictions.append(np.clip(estimate, 1, 5))
        return np.array(predictions)

