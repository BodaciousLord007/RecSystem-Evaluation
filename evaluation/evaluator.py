import time
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from .metrics import ranking_metrics, diversity


def eligible_users(heldout, threshold, limit, seed):
    # Recall needs at least one held-out liked book per evaluated user.
    users = np.sort(heldout.loc[heldout.rating >= threshold, 'user'].unique())
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(users, min(limit, len(users)), replace=False))


def evaluate(model, train, heldout, users, features, config, excluded=None):
    start = time.perf_counter()
    excluded = train if excluded is None else excluded
    seen = excluded.groupby('user').item.apply(set).to_dict()
    relevance = heldout[heldout.rating >= config.like_threshold].groupby('user').item.apply(set).to_dict()
    rows, diversities, catalog, examples = [], [], set(), []
    for user in users:
        # Rank the full unseen catalog, using the same candidates for every model.
        candidates = np.array([i for i in range(model.n_items) if i not in seen.get(user, set())])
        scores = model.score(int(user), candidates)
        if not np.all(np.isfinite(scores)):
            raise ValueError(f'Non-finite scores for user {user}')
        # Stable item-id tie breaking makes equal-score runs reproducible.
        top = candidates[np.lexsort((candidates, -scores))[:config.k]]
        rows.append(ranking_metrics(top, relevance.get(user, set()), config.k))
        diversities.append(diversity(top, features))
        catalog.update(top.tolist())
        if len(examples) < 3:
            examples.append({'user': int(user), 'items': top.tolist()})
    if not rows:
        raise ValueError('No users with held-out likes; increase the sample size.')
    # Give each evaluated user equal weight, regardless of history length.
    result = {f'{name}@{config.k}': float(np.mean([row[name] for row in rows]))
              for name in ('precision', 'recall', 'ndcg')}
    finite_diversities = [value for value in diversities if np.isfinite(value)]
    result.update(catalog_coverage=len(catalog) / model.n_items,
                  content_diversity=float(np.mean(finite_diversities)) if finite_diversities else np.nan,
                  evaluated_users=len(users),
                  eligible_user_fraction=len(relevance) / model.n_users,
                  ranking_seconds=time.perf_counter() - start)
    # Rating and classification metrics use observed pairs, not unrated books.
    observed = heldout[heldout.user.isin(users)]
    predictions = model.predict(observed.user.to_numpy(), observed.item.to_numpy())
    if model.rating_model:
        residual = observed.rating.to_numpy() - predictions
        result.update(rmse=float(np.sqrt(np.mean(residual ** 2))), mae=float(np.mean(np.abs(residual))))
    elif model.__class__.__name__ != 'Popularity':
        labels = (observed.rating.to_numpy() >= config.like_threshold).astype(int)
        result['f1_observed'] = float(f1_score(labels, predictions >= .5, zero_division=0))
        result['roc_auc_observed'] = float(roc_auc_score(labels, predictions)) if len(np.unique(labels)) == 2 else np.nan
    return result, examples

