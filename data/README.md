# Dataset

Goodbooks-10k: https://github.com/zygmuntz/goodbooks-10k

The downloader retrieves ratings.csv, books.csv, book_tags.csv and tags.csv from
that repository. Original dataset license: https://github.com/zygmuntz/goodbooks-10k/blob/master/LICENSE
Retain the source's attribution and license when redistributing data.
The downloaded files are ignored by git. SHA-256 checksums and URLs are recorded
in data/raw/manifest.json and copied into every results/run_metadata.json.

The source has about six million ratings on ten thousand books. Ratings use
book_id; book_tags uses goodreads_book_id. features.py joins using the latter.
No rating timestamps exist, so this project uses random per-user splits.

