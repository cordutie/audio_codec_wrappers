"""Codec registry. Static CodecInfo is available without importing any codec library (design D3)."""
from typing import Callable

import torch

from .base import CodecInfo, CodecWrapper

_INFOS: dict[str, CodecInfo] = {}
_FACTORIES: dict[str, Callable[..., CodecWrapper]] = {}
_CACHE: dict[tuple[str, str], CodecWrapper] = {}


def register(name: str, info: CodecInfo, factory: Callable[..., CodecWrapper]) -> None:
    _INFOS[name] = info
    _FACTORIES[name] = factory


def _unknown(name: str) -> KeyError:
    return KeyError(f"unknown codec {name!r}; known codecs: {list_codecs()}")


def list_codecs() -> list[str]:
    return sorted(_INFOS)


def codec_info(name: str) -> CodecInfo:
    if name not in _INFOS:
        raise _unknown(name)
    return _INFOS[name]


def _device_key(device) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def load_codec(name: str, device=None, **kwargs) -> CodecWrapper:
    """Load a wrapper, cached per (name, device) when no extra kwargs are given."""
    if name not in _FACTORIES:
        raise _unknown(name)
    dev = _device_key(device)
    if kwargs:
        return _FACTORIES[name](dev, **kwargs)
    key = (name, str(dev))
    if key not in _CACHE:
        _CACHE[key] = _FACTORIES[name](dev)
    return _CACHE[key]


# Lazy factories: the codec library is imported only when the codec is loaded.
def _lazy(module: str, cls: str) -> Callable[..., CodecWrapper]:
    def factory(device, **kwargs):
        import importlib
        mod = importlib.import_module(f"audio_codec_wrappers.codecs.{module}")
        return getattr(mod, cls)(device, **kwargs)
    return factory


def _toy(name: str):
    def factory(device, **kwargs):
        from .codecs.toy import ToyCodec
        return ToyCodec(_INFOS[name], device)
    return factory


# name, info, factory (CodecInfo values from plan.md Interfaces table)
register("encodec24k", CodecInfo("encodec24k", 24000, 1, 320, 128, 1, True, 0, False, False, "torch"),
         _lazy("encodec24", "Encodec24Codec"))
register("spectrostream", CodecInfo("spectrostream", 48000, 2, 1920, 256, 1, True, 1, False, False, "jax"),
         _lazy("spectrostream", "SpectroStreamCodec"))
register("encodec48k", CodecInfo("encodec48k", 48000, 2, 320, 128, 1, False, None, True, False, "torch"),
         _lazy("encodec48", "Encodec48Codec"))
register("music2latent", CodecInfo("music2latent", 44100, 2, 4096, 128, 1, False, None, False, True, "torch"),
         _lazy("music2latent", "Music2LatentWrapper"))
register("toy_causal", CodecInfo("toy_causal", 8000, 1, 8, 16, 1, True, 0, False, False, "torch"), _toy("toy_causal"))
register("toy_noncausal", CodecInfo("toy_noncausal", 8000, 1, 8, 16, 1, False, None, True, False, "torch"),
         _toy("toy_noncausal"))
