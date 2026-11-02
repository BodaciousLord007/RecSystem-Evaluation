import numpy as np
import pandas as pd


def prepare_data(directory, config):
    ratings = pd.read_csv(directory / 'ratings.csv')
    books = pd.read_csv(directory / 'books.csv')
    ratings = ratings.dropna(subset=['user_id', 'book_id', 'rating'])
    ratings = ratings.drop_duplicates(['user_id', 'book_id'])
    ratings = ratings[ratings.rating.between(1, 5)]
    # Fix the catalog by its metadata order, not by test-set rating popularity.
    books = books.drop_duplicates('book_id').sort_values('book_id').head(config.max_items)
    ratings = ratings[ratings.book_id.isin(books.book_id)]
    # Keep enough history for all three splits.
    eligible = ratings.groupby('user_id').size()
    eligible = eligible[eligible >= config.min_ratings].index.to_numpy()
    rng = np.random.default_rng(config.seed)
    users = np.sort(rng.choice(eligible, min(config.max_users, len(eligible)), replace=False))
    if len(users) == 0:
        raise ValueError('No eligible users. Increase --max-items or lower min_ratings.')
    ratings = ratings[ratings.user_id.isin(users)].copy()
    # Sparse matrices need consecutive indices; original IDs are saved separately.
    user_map = {value: i for i, value in enumerate(users)}
    item_map = {value: i for i, value in enumerate(books.book_id)}
    ratings['user'] = ratings.user_id.map(user_map).astype(int)
    ratings['item'] = ratings.book_id.map(item_map).astype(int)
    books = books.reset_index(drop=True)
    books['item'] = np.arange(len(books))
    # print('Mapped users/items:', len(user_map), len(item_map))
    return ratings[['user', 'item', 'rating']], books, users

