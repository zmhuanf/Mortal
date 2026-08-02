import os

import toml


def _deep_merge(base, extra):
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


config_file = os.environ.get('MORTAL_CFG', 'config.toml')
config = {}
with open(config_file, encoding='utf-8') as f:
    main = toml.load(f)
for path in main.pop('include', []):
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(config_file), path)
    with open(path, encoding='utf-8') as f:
        _deep_merge(config, toml.load(f))
_deep_merge(config, main)
