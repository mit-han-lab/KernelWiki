"""Configuration is data, never executable model configuration."""
from pathlib import Path
from .store import read


def merge(base, override):
    result = dict(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(root, config_path=None):
    config = read(root / 'data/repo-import.yaml')
    if config_path:
        config = merge(config, read(Path(config_path).resolve()))
    if config.get('version') != 1:
        raise ValueError('Unsupported configuration version')
    from .selection import KINDS
    kinds = config.get('scan', {}).get('entry_kinds', sorted(KINDS))
    if not isinstance(kinds, list) or any(kind not in KINDS for kind in kinds):
        raise ValueError('Unsupported scan.entry_kinds')
    enabled = config.get('selection', {}).get('enabled', True)
    if not isinstance(enabled, bool):
        raise ValueError('selection.enabled must be a boolean')
    return config


def load_preset(root, name, config):
    presets = read(root / config['presets_path']).get('presets', {})
    if name not in presets:
        raise ValueError(f'Unknown preset: {name}')
    preset = presets[name]
    if not preset.get('revision') or not preset.get('decision_manifest'):
        raise ValueError('Preset must pin a revision and provide decisions')
    return preset
