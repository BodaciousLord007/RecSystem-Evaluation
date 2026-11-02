import numpy as np


def ranking_metrics(recommended, relevant, k):
    relevant = set(relevant)
    hit = np.array([int(item in relevant) for item in recommended[:k]])
    precision = hit.sum() / k
    recall = hit.sum() / len(relevant) if relevant else np.nan
    # Hits near the top matter more; normalize against the best possible list.
    discounts = 1 / np.log2(np.arange(len(hit)) + 2)
    ideal = (1 / np.log2(np.arange(min(k, len(relevant))) + 2)).sum()
    ndcg = float(hit @ discounts / ideal) if ideal else np.nan
    return {'precision': precision, 'recall': recall, 'ndcg': ndcg}


def diversity(recommended, features):
    if len(recommended) < 2:
        return np.nan
    similarity = (features[recommended] @ features[recommended].T).toarray()
    # Count each pair once and leave out self-similarity.
    return float((1 - similarity)[np.triu_indices(len(recommended), 1)].mean())

