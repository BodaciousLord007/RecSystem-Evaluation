from sklearn.naive_bayes import MultinomialNB
from .common import ContentClassifier


class ContentNaiveBayes(ContentClassifier):
    def make_estimator(self):
        # Laplace smoothing keeps unseen features from forcing zero probability.
        return MultinomialNB(alpha=1.0)

