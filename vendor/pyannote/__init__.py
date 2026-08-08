"""Minimal stand-in for the ``pyannote`` package.

The remote GigaAM modeling file imports ``pyannote`` inside a lazily-called
helper (``get_pipeline``, only used by ``transcribe_longform``). transformers'
``check_imports`` still requires every top-level import to be importable
before the module can be used, so without this package the model would refuse
to load. This app never calls ``transcribe_longform``, so a package that
merely imports is sufficient.
"""
