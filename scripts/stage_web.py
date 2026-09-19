#!/usr/bin/env python3
"""Copy the built browser application into the generated Python package assets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

MARKER = '.neuroterrarium-generated.json'


def inventory(directory: Path) -> dict[str, str]:
    files = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('Generated application assets cannot contain symlinks')
        if path.is_file() and path.name != MARKER:
            files[path.relative_to(directory).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif not path.is_dir() and path.name != MARKER:
            raise ValueError('Unsupported generated asset type')
    return files


def stage(source: Path, target: Path) -> dict:
    source, target = Path(source), Path(target)
    if source.is_symlink() or target.is_symlink() or not source.is_dir():
        raise ValueError('Expected ordinary generated asset directories')
    if source.resolve() == target.resolve() or source.resolve() in target.resolve().parents or target.resolve() in source.resolve().parents:
        raise ValueError('Source and generated destination must be separate')
    files = inventory(source)
    if 'index.html' not in files or not any(name.endswith('.js') for name in files):
        raise ValueError('Build the browser application before staging it')
    if (source / MARKER).exists():
        raise ValueError('Unexpected marker in browser build output')
    if target.exists():
        marker = target / MARKER
        if not marker.is_file() or marker.is_symlink():
            raise ValueError('Refusing to replace an unmarked asset directory')
        old = json.loads(marker.read_text())
        if old.get('schema') != 'neuroterrarium.generated-static.v1' or old.get('files') != inventory(target):
            raise ValueError('Generated asset directory contains unrecognized or modified files')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.static-stage-', dir=target.parent) as temporary:
        staged = Path(temporary) / 'static'
        shutil.copytree(source, staged)
        if inventory(staged) != files:
            raise ValueError('Browser build changed while staging')
        result = {'schema': 'neuroterrarium.generated-static.v1', 'files': files}
        (staged / MARKER).write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
        if target.exists():
            # Only an unchanged directory marked by this function is removed.
            shutil.rmtree(target)
        staged.replace(target)
    return {'status': 'completed', 'files': len(files), 'index_sha256': files['index.html']}


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(stage(root / 'web/dist', root / 'src/neuroterrarium/static')))
