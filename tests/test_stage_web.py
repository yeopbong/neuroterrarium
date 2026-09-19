import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('stage_web', Path(__file__).parents[1] / 'scripts/stage_web.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def assets(tmp_path):
    source = tmp_path / 'dist'
    source.mkdir()
    (source / 'index.html').write_text('<script src="app.js"></script>')
    (source / 'app.js').write_text('export const version=1;')
    return source, tmp_path / 'package/static'


def test_staging_is_idempotent_and_removes_only_previous_generated_files(tmp_path):
    source, target = assets(tmp_path)
    first = module.stage(source, target)
    assert module.stage(source, target) == first
    (source / 'app.js').unlink()
    (source / 'new.js').write_text('export const version=2;')
    module.stage(source, target)
    assert not (target / 'app.js').exists()
    assert (target / 'new.js').is_file()
    assert not (target / 'dist').exists()


def test_unknown_or_modified_destination_is_preserved(tmp_path):
    source, target = assets(tmp_path)
    target.mkdir(parents=True)
    private = target / 'unrelated.txt'
    private.write_text('preserve')
    with pytest.raises(ValueError, match='unmarked'):
        module.stage(source, target)
    assert private.read_text() == 'preserve'
    private.unlink()
    target.rmdir()
    module.stage(source, target)
    private.write_text('preserve')
    with pytest.raises(ValueError, match='unrecognized'):
        module.stage(source, target)
    assert private.read_text() == 'preserve'


def test_symlinks_and_incomplete_build_fail(tmp_path):
    source, target = assets(tmp_path)
    (source / 'app.js').unlink()
    with pytest.raises(ValueError, match='Build'):
        module.stage(source, target)
    (source / 'app.js').symlink_to(source / 'index.html')
    with pytest.raises(ValueError, match='symlinks'):
        module.stage(source, target)
