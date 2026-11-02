import numpy as np
from scipy.sparse import csr_matrix


class BaseRecommender:
    # Only star-rating predictors should be evaluated with RMSE and MAE.
    rating_model = False

    def fit(self, ratings, n_users, n_items, features=None):
        self.n_users, self.n_items = n_users, n_items
        self.global_mean = float(ratings.rating.mean())
        # Missing entries stay implicit; zero is not an observed rating.
        self.matrix = csr_matrix((ratings.rating.to_numpy(dtype=float),
                                  (ratings.user.to_numpy(), ratings.item.to_numpy())),
                                 shape=(n_users, n_items))
        # Use the training mean if a user has no observations.
        self.user_mean = np.full(n_users, self.global_mean)
        means = ratings.groupby('user').rating.mean()
        self.user_mean[means.index] = means.to_numpy()
        counts = np.bincount(ratings.item, minlength=n_items)
        sums = np.bincount(ratings.item, weights=ratings.rating, minlength=n_items)
        # Ten pseudo-ratings pull rarely rated books toward the global mean.
        self.item_prior = (sums + 10 * self.global_mean) / (counts + 10)
        self.features = features
        return self

    def score(self, user, items):
        raise NotImplementedError

    def predict(self, users, items):
        return np.array([self.score(int(u), np.array([i]))[0] for u, i in zip(users, items)])

