import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

BASE_URL = 'https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/'
FILES = ('ratings.csv', 'books.csv', 'book_tags.csv', 'tags.csv')


def download_data(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in FILES:
        path = directory / name
        if not path.exists():
            print(f'Downloading {name} ...', flush=True)
            # An interrupted download must not look like a complete CSV.
            temporary = path.with_suffix('.part')
            with urlopen(BASE_URL + name, timeout=120) as response, temporary.open('wb') as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
            temporary.replace(path)
        # Record the exact data version used, including cached files.
        digest = hashlib.sha256()
        with path.open('rb') as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        manifest[name] = {'url': BASE_URL + name, 'sha256': digest.hexdigest()}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest

