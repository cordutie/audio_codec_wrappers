"""audio_codec_wrappers: one API for several neural audio codecs.

Importing the package must not import jax (checked by tests/test_env.py); it only sets the
environment so that jax shares the GPU with torch when it is imported later.
"""
import os


def configure_jax_env() -> None:
    """Disable XLA preallocation (jax would reserve 75% of the GPU at first use). Call before importing jax."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # No cuDNN/cuBLAS autotuning: with it, the first call per input shape took 10-80 s and held 8.5 GiB of GPU memory
    # (slice A2 measurement); without it 0.6 s and 0.3 GiB. Read by XLA when the backend starts.
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_gpu_autotune_level" not in flags:
        os.environ["XLA_FLAGS"] = (flags + " --xla_gpu_autotune_level=0").strip()


configure_jax_env()

from .base import CodecInfo, CodecWrapper, Encoded, StreamingDecoder  # noqa: E402
from .registry import codec_info, list_codecs, load_codec, register  # noqa: E402
