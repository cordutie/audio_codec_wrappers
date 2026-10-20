"""audio_codec_wrappers: one API for several neural audio codecs.

Importing the package must not import jax (checked by tests/test_env.py); it only sets the
environment so that jax shares the GPU with torch when it is imported later.
"""
import os


def configure_jax_env() -> None:
    """Disable XLA preallocation (jax would reserve 75% of the GPU at first use). Call before importing jax."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


configure_jax_env()
