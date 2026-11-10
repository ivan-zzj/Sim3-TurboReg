#!/usr/bin/env python3
"""Verify and safely unpack the official Stanford archives; never overwrite raw data."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ASSETS = {
    'bunny': ('bunny.tar.gz', 'https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz',
              'a5720bd96d158df403d153381b8411a727a1d73cff2f33dc9b212d6f75455b84',
              'bunny/reconstruction/bun_zipper.ply'),
    'armadillo': ('Armadillo.ply.gz', 'https://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz',
                  '8b9b56cc36e66d54429b1e1e75bd89e833645bfe0dc7c1afd1205877a7356a3f', 'Armadillo.ply'),
    'dragon': ('dragon_recon.tar.gz', 'https://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz',
               '74ac1d90989c9b1732edee82d57e9ce71452144cf4355f108d8c9c616d28d02f',
               'dragon_recon/dragon_vrip_res2.ply'),
    'buddha': ('happy_recon.tar.gz', 'https://graphics.stanford.edu/pub/3Dscanrep/happy/happy_recon.tar.gz',
               '409cd294efbfd8244e15a382b95a9423f153b7776e736c9b09f19ec9d3c10ed0',
               'happy_recon/happy_vrip_res2.ply'),
}


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root / 'stanford'
    dest = root / 'extracted'
    dest.mkdir(exist_ok=True)
    records = {}
    for name, (archive, url, expected, selected) in ASSETS.items():
        path = root / archive
        if sha256(path) != expected:
            raise ValueError(f'Archive changed: {path}')
        with gzip.open(path, 'rb') as f:
            while f.read(1024 * 1024):
                pass  # read through EOF to verify gzip CRC and truncation
        if archive.endswith('.tar.gz'):
            with tarfile.open(path) as tf:
                members = tf.getmembers()
                for m in members:
                    relative = Path(m.name)
                    if relative.is_absolute() or '..' in relative.parts or not (m.isfile() or m.isdir()):
                        raise ValueError(f'Unsafe archive member: {m.name}')
                    target = dest / relative
                    if not target.resolve().is_relative_to(dest.resolve()):
                        raise ValueError(f'Unsafe existing extraction path: {target}')
                for m in members:
                    target = dest / m.name
                    if m.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        contents = tf.extractfile(m).read()
                        if target.exists():
                            if target.read_bytes() != contents:
                                raise ValueError(f'Refusing to replace modified file: {target}')
                        else:
                            target.write_bytes(contents)
        else:
            target = dest / selected
            with gzip.open(path, 'rb') as f:
                contents = f.read()
            if target.exists() and target.read_bytes() != contents:
                raise ValueError(f'Refusing to replace modified file: {target}')
            if not target.exists():
                target.write_bytes(contents)
        ply = dest / selected
        records[name] = dict(archive=archive, url=url, archive_sha256=expected,
                             archive_bytes=path.stat().st_size, gzip_crc_verified=True,
                             selected_ply=str(ply.relative_to(args.root)), ply_sha256=sha256(ply))
        print(name, records[name], flush=True)
    manifest = dict(source='https://graphics.stanford.edu/data/3Dscanrep/',
                    hash_note='Locally pinned hashes, not publisher-provided checksums.',
                    reconstruction_note='Paper does not identify exact PLY resolutions; selected highest resolution in each downloaded archive.',
                    objects=records)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    papers = args.root / 'papers'
    papers.mkdir(exist_ok=True)
    for source, target in [('/tmp/teaser_original.pdf', 'teaser_2001.07715v2.pdf'),
                           ('/tmp/teaser_raw.txt', 'teaser_2001.07715v2.txt'),
                           ('/tmp/teaser_master_tree.json', 'official_master_tree.json'),
                           ('/tmp/teaser_develop_tree.json', 'official_develop_tree.json'),
                           ('/tmp/teaser_branches.json', 'official_branches.json')]:
        shutil.copyfile(source, papers / target)
    (papers / 'manifest.json').write_text(json.dumps(dict(
        paper_version='arXiv:2001.07715v2, 17 October 2020 (verified printed first-page identifier)',
        paper_url='https://arxiv.org/pdf/2001.07715v2',
        paper_sha256=sha256(papers / 'teaser_2001.07715v2.pdf'),
        official_commit='52a9c52ee7d4c838c5e8a75458c33178be5bfb70'), indent=2) + '\n')


if __name__ == '__main__':
    main()
