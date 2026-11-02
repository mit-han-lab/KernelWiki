"""Persistent records and recoverable writes."""
import hashlib
import json
import os
import re
from pathlib import Path

from _yaml_compat import yaml


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def valid_receipt(path, receipt):
    """Missing bytes and a missing digest must never count as matching evidence."""
    checksum, locator = receipt.get('sha256'), receipt.get('locator')
    return (isinstance(checksum, str) and re.fullmatch(r'[0-9a-f]{64}', checksum) is not None
            and isinstance(locator, str) and bool(locator.strip())
            and path.is_file() and file_hash(path) == checksum)


def read(path, default=None):
    if not path.exists():
        return {} if default is None else default
    text = path.read_text()
    loader = getattr(yaml, 'CSafeLoader', None)
    value = yaml.load(text, Loader=loader) if loader is not None else yaml.safe_load(text)
    return value or ({} if default is None else default)


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write(path, value):
    atomic_write(path, yaml.safe_dump(value, sort_keys=False))


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f'Path escapes root: {relative}')
    return path
