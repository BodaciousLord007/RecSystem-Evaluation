import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize


def build_features(books, directory, use_tags=False):
    # Do not include average_rating, ratings_count or other aggregated outcomes.
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000,
                                token_pattern=r'(?u)\b\w+\b')
    titles = vectorizer.fit_transform(books.original_title.fillna(books.title).fillna('unknown'))
    authors = MultiLabelBinarizer(sparse_output=True).fit_transform(
        books.authors.fillna('unknown').str.split(', ').tolist())
    language = pd.get_dummies(books.language_code.fillna('unknown'), dtype=float)
    year = pd.to_numeric(books.original_publication_year, errors='coerce').fillna(0)
    # Coarse buckets avoid treating publication year as a large numerical weight.
    decades = pd.get_dummies((year.clip(0, 2030) // 10).astype(int), dtype=float)
    blocks = [titles, authors, csr_matrix(language.to_numpy()), csr_matrix(decades.to_numpy())]
    if use_tags:
        # Snapshot tags are collective user annotations with no event timestamps.
        # This optional experiment is transductive; the default avoids that caveat.
        tags = pd.read_csv(directory / 'tags.csv')
        links = pd.read_csv(directory / 'book_tags.csv')
        links = links[links.goodreads_book_id.isin(books.goodreads_book_id)]
        links = links.merge(tags, on='tag_id').sort_values('count', ascending=False)
        links = links.groupby('goodreads_book_id', sort=False).head(20)
        text = links.groupby('goodreads_book_id').tag_name.apply(lambda values: ' '.join(values))
        docs = books.goodreads_book_id.map(text).fillna('')
        if docs.str.strip().ne('').any():
            blocks.append(TfidfVectorizer(max_features=2000, stop_words='english').fit_transform(docs))
    features = normalize(hstack(blocks, format='csr')).astype(np.float64)
    # print('Content matrix:', features.shape, 'nonzero entries:', features.nnz)
    return features

