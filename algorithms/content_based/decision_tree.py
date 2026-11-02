from sklearn.tree import DecisionTreeClassifier
from .common import ContentClassifier


class ContentDecisionTree(ContentClassifier):
    def make_estimator(self):
        # Limit depth because each user's training set is small.
        return DecisionTreeClassifier(criterion='entropy', max_depth=5,
                                      min_samples_leaf=3, random_state=self.seed)

