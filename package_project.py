from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def package():
    root = Path(__file__).resolve().parent
    files = [root / name for name in (
        'README.md', 'requirements.txt', 'config.py', 'run_experiments.py',
        'package_project.py', 'data/README.md', 'report/report.md')]
    for folder in ('algorithms', 'preprocessing', 'evaluation', 'tests', 'notebooks'):
        files.extend(path for path in (root / folder).rglob('*')
                     if path.is_file() and path.suffix in ('.py', '.ipynb'))
    target = root / 'goodbooks_project.zip'
    with ZipFile(target, 'w', ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    print(f'Created {target} ({target.stat().st_size:,} bytes)')


if __name__ == '__main__':
    package()
