import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
from config import Config
from preprocessing.download import download_data
from preprocessing.prepare_data import prepare_data
from preprocessing.split import split_ratings
from preprocessing.features import build_features
from algorithms.popularity import Popularity
from algorithms.collaborative.user_knn import UserKNN
from algorithms.collaborative.item_knn import ItemKNN
from algorithms.collaborative.matrix_factorization import MatrixFactorization
from algorithms.content_based.decision_tree import ContentDecisionTree
from algorithms.content_based.naive_bayes import ContentNaiveBayes
from algorithms.content_based.content_knn import ContentKNN
from evaluation.evaluator import evaluate, eligible_users
from evaluation.plots import plot_results

ROOT = Path(__file__).resolve().parent
MODEL_NAMES = ('popularity', 'user_knn', 'item_knn', 'matrix_factorization',
               'decision_tree', 'naive_bayes', 'content_knn')


def make_model(name, config, neighbors=None):
    k = config.neighbors if neighbors is None else neighbors
    return {
        'popularity': lambda: Popularity(config.like_threshold),
        'user_knn': lambda: UserKNN(k),
        'item_knn': lambda: ItemKNN(k),
        'matrix_factorization': lambda: MatrixFactorization(config.factors, config.epochs, config.seed),
        'decision_tree': lambda: ContentDecisionTree(config.like_threshold, config.seed),
        'naive_bayes': lambda: ContentNaiveBayes(config.like_threshold, config.seed),
        'content_knn': lambda: ContentKNN(k, config.like_threshold),
    }[name]()


def main():
    parser = argparse.ArgumentParser(description='Compare collaborative and content based book recommenders.')
    parser.add_argument('--model', choices=('all',) + MODEL_NAMES, default='all')
    parser.add_argument('--max-users', type=int, default=500)
    parser.add_argument('--max-items', type=int, default=2000)
    parser.add_argument('--eval-users', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--use-tags', action='store_true')
    parser.add_argument('--tune', action='store_true', help='Select kNN neighbourhood size on validation NDCG')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    if min(args.max_users, args.eval_users, args.epochs, args.k) < 1 or not 10 <= args.max_items <= 10000:
        parser.error('Sizes must be positive; max-items must be between 10 and 10000.')
    config = Config(max_users=args.max_users, max_items=args.max_items,
                    eval_users=args.eval_users, epochs=args.epochs, seed=args.seed, k=args.k)
    raw, processed = ROOT / 'data/raw', ROOT / 'data/processed'
    manifest = download_data(raw)
    ratings, books, original_users = prepare_data(raw, config)
    train, validation, test = split_ratings(ratings, config.seed)
    features = build_features(books, raw, args.use_tags)
    processed.mkdir(parents=True, exist_ok=True)
    for name, frame in [('train', train), ('validation', validation), ('test', test)]:
        frame.to_csv(processed / f'{name}.csv', index=False)
    books[['item', 'book_id', 'goodreads_book_id', 'title']].to_csv(processed / 'item_mapping.csv', index=False)
    pd.DataFrame({'user': np.arange(len(original_users)), 'original_user_id': original_users}).to_csv(processed / 'user_mapping.csv', index=False)
    # Fix the evaluation users once so model comparisons stay fair.
    users = eligible_users(test, config.like_threshold, config.eval_users, config.seed)
    final_train = pd.concat([train, validation], ignore_index=True)
    print(f'Data: {len(original_users)} users, {len(books)} books, {len(ratings)} ratings; {len(users)} test users', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    names = MODEL_NAMES if args.model == 'all' else [args.model]
    results, selections, examples, tuning = [], {}, {}, []
    for name in names:
        print(f'Training {name} ...', flush=True)
        chosen = config.neighbors
        # Choose neighbours on validation data; test scores never guide selection.
        if args.tune and name in ('user_knn', 'item_knn', 'content_knn'):
            val_users = eligible_users(validation, config.like_threshold, config.eval_users, config.seed)
            best = -np.inf
            for neighbors in (10, 30, 50):
                candidate = make_model(name, config, neighbors).fit(train, len(original_users), len(books), features)
                metrics, _ = evaluate(candidate, train, validation, val_users, features, config)
                value = metrics[f'ndcg@{config.k}']
                tuning.append({'model': name, 'neighbors': neighbors, 'validation_ndcg': value})
                if value > best:
                    best, chosen = value, neighbors
        selections[name] = {'neighbors': chosen} if 'knn' in name else {'preset': 'see source/config.py'}
        start = time.perf_counter()
        # Refit with validation ratings after fixing the hyperparameters.
        model = make_model(name, config, chosen).fit(final_train, len(original_users), len(books), features)
        seconds = time.perf_counter() - start
        metrics, samples = evaluate(model, final_train, test, users, features, config)
        results.append({'model': name, 'fit_seconds': seconds, **metrics})
        examples[name] = [{'user': row['user'], 'books': books.iloc[row['items']].title.tolist()} for row in samples]
        # print('First recommendation list:', examples[name][0])
        print(f"  NDCG@{config.k}={metrics[f'ndcg@{config.k}']:.4f}", flush=True)
        del model
    frame = pd.DataFrame(results)
    frame.to_csv(args.output / 'metrics.csv', index=False)
    plot_results(frame, args.output / 'comparison.png', config.k)
    (args.output / 'recommendations.json').write_text(json.dumps(examples, indent=2), encoding='utf-8')
    if tuning:
        pd.DataFrame(tuning).to_csv(args.output / 'validation.csv', index=False)
    metadata = {'config': asdict(config), 'use_tags': args.use_tags, 'tuned': args.tune,
                'selection': selections, 'data': manifest, 'test_users': users.tolist(),
                'split_sizes': {'train': len(train), 'validation': len(validation), 'test': len(test)},
                'python': platform.python_version(),
                'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib']}}
    (args.output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(frame.to_string(index=False))
    print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()

