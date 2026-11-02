import numpy as np
from ..base import BaseRecommender


class MatrixFactorization(BaseRecommender):
    rating_model = True

    def __init__(self, factors=24, epochs=20, seed=42, learning_rate=.03, regularization=.05):
        self.factors, self.epochs, self.seed = factors, epochs, seed
        self.lr, self.reg = learning_rate, regularization

    def fit(self, ratings, n_users, n_items, features=None):
        super().fit(ratings, n_users, n_items, features)
        rng = np.random.default_rng(self.seed)
        self.p = rng.normal(0, .1, (n_users, self.factors))
        self.q = rng.normal(0, .1, (n_items, self.factors))
        # Biases capture generous users and books that usually score well.
        self.bu, self.bi = np.zeros(n_users), np.zeros(n_items)
        users, items, values = ratings.user.to_numpy(), ratings.item.to_numpy(), ratings.rating.to_numpy()
        for epoch in range(self.epochs):
            order = rng.permutation(len(ratings))
            for start in range(0, len(order), 256):
                idx = order[start:start + 256]
                u, i = users[idx], items[idx]
                # Both factor updates must use the values from before this step.
                pu, qi = self.p[u].copy(), self.q[i].copy()
                error = values[idx] - (self.global_mean + self.bu[u] + self.bi[i] + (pu * qi).sum(axis=1))
                # Average repeated-index updates so popular books cannot explode a batch.
                uc = np.bincount(u, minlength=n_users)[u]
                ic = np.bincount(i, minlength=n_items)[i]
                np.add.at(self.bu, u, self.lr * (error - self.reg * self.bu[u]) / uc)
                np.add.at(self.bi, i, self.lr * (error - self.reg * self.bi[i]) / ic)
                np.add.at(self.p, u, self.lr * (error[:, None] * qi - self.reg * pu) / uc[:, None])
                np.add.at(self.q, i, self.lr * (error[:, None] * pu - self.reg * qi) / ic[:, None])
            # print('Epoch and last-batch RMSE:', epoch + 1, np.sqrt(np.mean(error ** 2)))
        # Unobserved catalog items retain a neutral baseline, not random factors.
        unseen = np.bincount(items, minlength=n_items) == 0
        self.q[unseen] = 0
        return self

    def score(self, user, items):
        items = np.asarray(items)
        return np.clip(self.global_mean + self.bu[user] + self.bi[items] + self.q[items] @ self.p[user], 1, 5)

