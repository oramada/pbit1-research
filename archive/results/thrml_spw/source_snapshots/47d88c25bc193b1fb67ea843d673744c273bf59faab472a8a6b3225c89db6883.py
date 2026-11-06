"""Content-preserving hard-link deduplication of completed research arrays."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import os
import shutil
import stat

ROOT = Path('results/thrml_spw')
OUT = ROOT/'identical_artifact_dedup_20261028'
EXCLUDE = ['/probability_cohort_evidence/', '/allocated_native_evidence/', '/allocated_statistics_evidence/']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def same_bytes(a, b):
    with a.open('rb') as left, b.open('rb') as right:
        while True:
            x, y = left.read(4*1024*1024), right.read(4*1024*1024)
            assert x == y, (a, b)
            if not x:
                break


def identity(path):
    s = path.lstat()
    assert stat.S_ISREG(s.st_mode) and not path.is_symlink() and s.st_uid == os.getuid()
    assert path.resolve().is_relative_to(ROOT.resolve())
    return {'device': s.st_dev, 'inode': s.st_ino, 'bytes': s.st_size, 'links': s.st_nlink,
            'mtime_ns': s.st_mtime_ns, 'mode': stat.S_IMODE(s.st_mode), 'uid': s.st_uid, 'gid': s.st_gid,
            'allocated_bytes': s.st_blocks*512}


def main():
    assert not (OUT/'inventory.json').exists()
    groups = collections.defaultdict(list)
    for base, dirs, files in os.walk(ROOT):
        for filename in files:
            path = Path(base)/filename
            if path.suffix not in ['.npz', '.npy'] or path.is_symlink() or any(p in str(path) for p in EXCLUDE):
                continue
            s = path.stat()
            if s.st_size >= 1000000 and s.st_uid == os.getuid():
                groups[s.st_size].append(path)
    plans, hashed = [], {}
    for size, paths in groups.items():
        if len({(p.stat().st_dev, p.stat().st_ino) for p in paths}) < 2:
            continue
        byhash = collections.defaultdict(list)
        for path in paths:
            meta = identity(path)
            key = meta['device'], meta['inode']
            if key not in hashed:
                hashed[key] = digest(path)
            byhash[hashed[key]].append((path, meta))
        for checksum, entries in byhash.items():
            inodes = collections.defaultdict(list)
            for path, meta in entries:
                inodes[meta['device'], meta['inode']].append((path, meta))
            if len(inodes) < 2:
                continue
            canonical, cm = max(entries, key=lambda pair: (pair[1]['links'], str(pair[0])))
            replacements, freed = [], 0
            for key, aliases in inodes.items():
                if key == (cm['device'], cm['inode']):
                    continue
                # Only replace an inode when all its hard links are in this inventory.
                if len(aliases) != aliases[0][1]['links']:
                    continue
                if any(any(meta[k] != cm[k] for k in ['device', 'mode', 'uid', 'gid']) for _, meta in aliases):
                    continue
                for path, meta in aliases:
                    same_bytes(canonical, path)
                    replacements.append({'path': str(path), 'before': meta})
                freed += aliases[0][1]['allocated_bytes']
            if replacements:
                plans.append({'canonical': str(canonical), 'canonical_before': cm, 'sha256': checksum,
                              'replacements': replacements, 'allocated_bytes_reclaimable': freed})
    assert plans
    before = shutil.disk_usage('.').free
    inventory = {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'source_sha256': digest(__file__), 'free_bytes_before': before, 'groups': plans,
                 'scope': 'Only byte-identical completed research NPY/NPZ copies become hard links. Paths/file contents/checksums stay unchanged; inode/link/ctime and canonical shared mtime may change. Original per-path metadata retained here. No source/config/results content, environment/cache/user data or active output tree is modified. Immutable linked evidence must not later be edited in place; future changed copies must use atomic replacement or break their link first. This is software disk deduplication, not model compression.'}
    save(OUT/'inventory.json', inventory)
    replaced = []
    for plan in plans:
        canonical = Path(plan['canonical'])
        cm = identity(canonical)
        assert (cm['device'], cm['inode'], cm['bytes'], cm['mtime_ns']) == tuple(plan['canonical_before'][k] for k in ['device', 'inode', 'bytes', 'mtime_ns'])
        assert digest(canonical) == plan['sha256']
        for row in plan['replacements']:
            path = Path(row['path'])
            actual = identity(path)
            assert all(actual[k] == row['before'][k] for k in ['device', 'inode', 'bytes', 'mtime_ns', 'mode', 'uid', 'gid'])
            same_bytes(canonical, path)
            temporary = path.with_name(path.name+'.dedup_link.tmp')
            assert not temporary.exists()
            os.link(canonical, temporary)
            os.replace(temporary, path)
            assert path.stat().st_ino == canonical.stat().st_ino
            replaced.append(str(path))
            save(OUT/'progress.json', {'replaced_paths': replaced})
        assert digest(canonical) == plan['sha256']
    for plan in plans:
        canonical = Path(plan['canonical'])
        assert digest(canonical) == plan['sha256']
        for row in plan['replacements']:
            actual = identity(Path(row['path']))
            assert actual['inode'] == canonical.stat().st_ino and actual['bytes'] == row['before']['bytes']
    save(OUT/'completion.json', {'completed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 'groups': len(plans), 'paths_linked': len(replaced),
                                 'allocated_file_bytes_reclaimed': sum(p['allocated_bytes_reclaimable'] for p in plans),
                                 'free_bytes_before': before, 'actual_free_bytes_after': shutil.disk_usage('.').free,
                                 'inventory_sha256': digest(OUT/'inventory.json'), 'source_sha256': digest(__file__),
                                 'all_file_contents_and_paths_preserved': True, 'scope': inventory['scope']})
    print((OUT/'completion.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
