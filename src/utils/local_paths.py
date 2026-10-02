"""Machine-local paths and strict, hash-backed media relocation verification."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil

ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_local_paths(path=None):
    path = Path(path) if path is not None else ROOT / 'configs/local_paths.yaml'
    if not path.is_file():
        return {}
    import yaml
    with path.open(encoding='utf-8-sig') as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError('Local paths configuration must be a mapping')
    return value


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def resolve_media_path(value, config=None):
    """Translate only registered POSIX root boundaries, leaving IDs unchanged."""
    config = load_local_paths() if config is None else config
    text = str(value)
    posix = PurePosixPath(text)
    if '..' in posix.parts or '..' in PureWindowsPath(text).parts:
        raise ValueError('Parent traversal is not a valid manifest media path')
    mappings = sorted(config.get('media_roots', []),
                      key=lambda entry: len(PurePosixPath(entry['source']).parts), reverse=True)
    for entry in mappings:
        source = PurePosixPath(entry['source'])
        if not source.is_absolute() or '..' in source.parts:
            raise ValueError('Source roots must be absolute POSIX paths')
        try:
            relative = posix.relative_to(source)
        except ValueError:
            continue
        target_root = project_path(entry['target']).resolve()
        target = target_root.joinpath(*relative.parts).resolve()
        if not target.is_relative_to(target_root):
            raise ValueError('Resolved media path escaped the configured root')
        return target
    if os.name == 'nt' and posix.is_absolute() and not PureWindowsPath(text).drive:
        raise ValueError(f'No local root mapping for Linux media path: {text}')
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def ffmpeg_executable():
    """Use an explicit binary, PATH, or the binary installed in this venv."""
    configured = os.environ.get('FFMPEG_EXE') or load_local_paths().get('ffmpeg')
    if configured:
        path = project_path(configured)
        if not path.is_file():
            raise FileNotFoundError(f'Configured FFmpeg missing: {path}')
        return str(path)
    found = shutil.which('ffmpeg')
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as error:
        raise FileNotFoundError('Install requirements-windows.txt or configure FFMPEG_EXE') from error


@lru_cache(maxsize=32768)
def _verified_hash(path, size, mtime_ns, ctime_ns):
    # Rehash whenever file metadata changes; cache is per Python process only.
    return file_sha256(path)


class MediaFileVerifier:
    """Native files retain size/mtime checks; moved files also require source SHA256."""
    def __init__(self, eligibility_path, config=None):
        self.config = load_local_paths() if config is None else config
        self.records = {}
        self.audit_sha256 = None
        audit_path = self.config.get('relocation_audit')
        if audit_path and project_path(audit_path).is_file():
            path = project_path(audit_path)
            self.audit_sha256 = file_sha256(path)
            with path.open(encoding='utf-8') as stream:
                header = json.loads(next(stream))
                if header.get('format') != 'media_relocation_v1':
                    raise ValueError('Unknown relocation audit format')
                if header['eligibility_sha256'] != file_sha256(eligibility_path):
                    raise ValueError('Relocation audit belongs to another eligibility file')
                for line in stream:
                    entry = json.loads(line)
                    key = (entry['dataset'], entry['sample_id'])
                    if key in self.records:
                        raise ValueError('Duplicate relocation record')
                    self.records[key] = entry

    def verify(self, row):
        path = Path(row['path'])
        stat = path.stat()
        source = row.get('source_path', row['path'])
        moved = str(path) != str(Path(source))
        entry = self.records.get((row['dataset'], row['sample_id']))
        if entry is None:
            if moved:
                raise ValueError('Relocated media requires a verified relocation audit: ' + str(path))
            if stat.st_size != row['size'] or stat.st_mtime_ns != row['mtime_ns']:
                raise ValueError('Media changed after quality audit')
            return
        if entry['source_path'] != source or Path(entry['local_path']).resolve() != path.resolve():
            raise ValueError('Relocation path mismatch')
        if (entry['source_size'] != row['size'] or entry['source_mtime_ns'] != row['mtime_ns']
                or entry['size'] != stat.st_size or entry['mtime_ns'] != stat.st_mtime_ns):
            raise ValueError('Media metadata differs from relocation audit')
        if entry['source_sha256'] != entry['sha256']:
            raise ValueError('Source and local content differ')
        actual = _verified_hash(str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if actual != entry['sha256']:
            raise ValueError('Relocated media hash mismatch')

    def provenance(self):
        return dict(policy='media_relocation_v1', audit_sha256=self.audit_sha256)
