# Bundled YAML fallback

The `yaml/` package in this directory is the pure-Python portion of PyYAML
6.0.3.  KernelWiki uses it only when the host Python cannot import its own
PyYAML installation, so the query and validation commands remain usable on a
clean Python installation and without network access.

The optional C extension is intentionally not included.  The bundled code is
loaded through `scripts/_yaml_compat.py`, which prefers the host installation
when one is available.

PyYAML is distributed under the MIT license; see `LICENSE`.
