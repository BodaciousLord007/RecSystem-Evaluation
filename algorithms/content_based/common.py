import numpy as np
from ..base import BaseRecommender


class ContentClassifier(BaseRecommender):
    def __init__(self, threshold=4, seed=42):
        self.threshold, self.seed = threshold, seed

    def fit(self, ratings, n_users, n_items, features=None):
        super().fit(ratings, n_users, n_items, features)
        self.models, self.priors = {}, {}
        # Each classifier learns only from the active user's own ratings.
        for user, group in ratings.groupby('user'):
            labels = (group.rating.to_numpy() >= self.threshold).astype(int)
            # Smoothed like rate handles users with only one observed class.
            self.priors[user] = (labels.sum() + 1) / (len(labels) + 2)
            if len(np.unique(labels)) == 2:
                model = self.make_estimator()
                model.fit(features[group.item.to_numpy()], labels)
                self.models[user] = model
            # print('User, observations, liked:', user, len(labels), labels.sum())
        return self

    def score(self, user, items):
        if user not in self.models:
            return np.full(len(items), self.priors.get(user, .5))
        model = self.models[user]
        # Rank by P(liked), not by the hard predicted label.
        column = list(model.classes_).index(1)
        return model.predict_proba(self.features[np.asarray(items)])[:, column]

