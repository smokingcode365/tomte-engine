"""Inspect built archives without extracting or importing their contents."""
import argparse
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


def inspect(path: Path) -> int:
    if path.suffix == '.whl':
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            assert any(name.endswith('.dist-info/METADATA') for name in names), 'Missing wheel metadata'
    elif path.name.endswith('.tar.gz'):
        with tarfile.open(path, 'r:gz') as archive:
            members = archive.getmembers()
            assert all(not member.issym() and not member.islnk() for member in members), 'Unexpected archive link'
            names = [member.name for member in members]
            assert any(name.endswith('/pyproject.toml') for name in names), 'Missing source project metadata'
    else:
        raise ValueError('Expected a wheel or source distribution')
    forbidden = {'.git', '.venv', '.cache', '__pycache__', 'prototypes',
                 '.private-diagnostics', 'private_reviews', 'AGENTS.md'}
    for name in names:
        parts = PurePosixPath(name).parts
        assert parts and not name.startswith('/') and '..' not in parts, 'Unsafe archive path'
        assert not forbidden.intersection(parts), 'Unexpected local/private archive member: ' + name
        root = parts[0] if path.suffix == '.whl' else (parts[1] if len(parts) > 1 else '')
        assert root != 'log', 'Unexpected generated log directory'
        assert all(part != '.env' and not (part.startswith('.env.') and part != '.env.example')
                   for part in parts), 'Unexpected environment file'
    assert any(name.endswith('tomte/engine/__init__.py') for name in names), 'Missing engine package'
    assert any(name.endswith('LICENSE') for name in names), 'Missing license'
    print(f'{path.name}: inspected {len(names)} members')
    return len(names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', nargs='+', type=Path)
    for archive in parser.parse_args().archives:
        inspect(archive)


if __name__ == '__main__':
    main()
