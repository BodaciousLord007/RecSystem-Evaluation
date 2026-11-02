import numpy as np
from ..base import BaseRecommender


class ContentKNN(BaseRecommender):
    def __init__(self, k=30, threshold=4):
        self.k, self.threshold = k, threshold

    def score(self, user, items):
        history = self.matrix.getrow(user)
        labels = (history.data >= self.threshold).astype(float)
        # Fall back to the user's like rate when no book shares content.
        prior = (labels.sum() + 1) / (len(labels) + 2)
        # Features are unit-normalized, so their dot product is cosine similarity.
        similarity = (self.features[np.asarray(items)] @ self.features[history.indices].T).toarray()
        result = []
        for weights in similarity:
            selected = np.flatnonzero(weights > 1e-12)
            selected = selected[np.argsort(weights[selected])[-self.k:]]
            # print('Content neighbours with nonzero similarity:', len(selected))
            # Similar books contribute more to the predicted like probability.
            result.append(np.dot(weights[selected], labels[selected]) / weights[selected].sum()
                          if len(selected) else prior)
        return np.array(result)

