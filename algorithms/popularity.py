import numpy as np
from .base import BaseRecommender


class Popularity(BaseRecommender):
    def __init__(self, threshold=4):
        self.threshold = threshold

    def fit(self, ratings, n_users, n_items, features=None):
        super().fit(ratings, n_users, n_items, features)
        self.counts = np.bincount(ratings.loc[ratings.rating >= self.threshold, 'item'],
                                  minlength=n_items)
        return self

    def score(self, user, items):
        return self.counts[np.asarray(items)]

