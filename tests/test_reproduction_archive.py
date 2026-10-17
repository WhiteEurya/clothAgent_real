import hashlib
import json
import sys
import tarfile
from io import BytesIO

import pytest
import zstandard

from scripts.package_reproduction import Parts
from scripts.restore_reproduction import JoinedParts, main


def test_split_writer_and_joined_reader_preserve_bytes(tmp_path):
    writer = Parts(tmp_path, limit=7)
    data = bytes(range(50))
    writer.write(data[:13])
    writer.write(data[13:])
    writer.finish()
    reader = JoinedParts([tmp_path/row['name'] for row in writer.rows])
    assert b''.join(iter(lambda: reader.read(9), b'')) == data
    for row in writer.rows:
        assert hashlib.sha256((tmp_path/row['name']).read_bytes()).hexdigest() == row['sha256']


@pytest.mark.parametrize('condition', ['normal', 'conflict', 'traversal', 'corrupt'])
def test_restore_checks_hashes_and_preserves_existing_data(tmp_path, monkeypatch, condition):
    root, cache, ssd = [tmp_path/n for n in ('root', 'cache', 'ssd')]
    (root/'reproduction').mkdir(parents=True)
    cache.mkdir()
    name = 'workspace/../escape' if condition == 'traversal' else 'workspace/runs/test.json'
    content = b'{"immutable": true}'
    tar = BytesIO()
    with tarfile.open(fileobj=tar, mode='w') as stream:
        info = tarfile.TarInfo(name)
        info.size = len(content)
        stream.addfile(info, BytesIO(content))
    archive = cache/'evidence.tar.zst.part000'
    archive.write_bytes(zstandard.ZstdCompressor().compress(tar.getvalue()))
    inventory = cache/'files.jsonl'
    inventory.write_text(json.dumps(dict(path=name, size=len(content), sha256=hashlib.sha256(content).hexdigest()))+'\n')
    def asset(path):
        return dict(name=path.name, size=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (root/'reproduction/manifest.json').write_text(json.dumps(dict(
        archive_parts=[asset(archive)], inventory=asset(inventory), download_base='https://invalid.example')))
    monkeypatch.setattr(sys, 'argv', ['restore', '--root', str(root), '--cache', str(cache), '--ssd-root', str(ssd)])
    if condition == 'conflict':
        (root/'runs').mkdir()
        (root/'runs/test.json').write_bytes(b'local data')
    if condition == 'corrupt':
        archive.write_bytes(b'broken')
    if condition == 'normal':
        main()
        assert (root/'runs/test.json').read_bytes() == content
        main()
    else:
        with pytest.raises(ValueError):
            main()
        if condition == 'conflict':
            assert (root/'runs/test.json').read_bytes() == b'local data'
        assert not (root/'escape').exists()
