"""Subprocess worker entrypoints (train, infer).

Each worker is its own `python -m` entry so it gets a fresh CUDA context
and its crash/OOM does not take down the API process.
"""
