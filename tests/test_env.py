"""Slice A0: the joint torch + jax + numpy-2 environment (design D1, D16, A1; brief RQ2 gate)."""
import os
import subprocess
import sys

import torch

from conftest import REPO


def test_torch():
    assert torch.__version__ == "2.10.0+cu128"
    assert torch.cuda.is_available()


def test_jax_gpu():
    import jax
    assert jax.__version__ == "0.10.1"
    assert len(jax.devices("gpu")) > 0


def test_numpy2():
    import numpy as np
    assert int(np.__version__.split(".")[0]) == 2


def test_prealloc_env():
    """Importing the package sets XLA_PYTHON_CLIENT_PREALLOCATE=false before jax is imported."""
    code = (
        "import os, sys\n"
        "import audio_codec_wrappers\n"
        "assert 'jax' not in sys.modules, 'importing the package must not import jax'\n"
        "print(os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'))\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "XLA_PYTHON_CLIENT_PREALLOCATE"}
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "false"


def test_dlpack_roundtrip():
    from audio_codec_wrappers.jax_bridge import jax_to_torch, torch_to_jax
    t = torch.arange(12.0).cuda()
    a = torch_to_jax(t)
    assert all(d.platform == "gpu" for d in a.devices())
    back = jax_to_torch(a)
    assert back.device == t.device
    assert torch.equal(back, t)


def test_shared_gpu_compute():
    import jax
    import jax.numpy as jnp
    gpu = jax.devices("gpu")[0]
    a = jax.device_put(jnp.array([[1.0, 2.0], [3.0, 4.0]]), gpu)
    b = jax.device_put(jnp.array([[5.0], [6.0]]), gpu)
    c = a @ b
    assert c.tolist() == [[17.0], [39.0]]
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], device="cuda")
    k = torch.tensor([[[1.0, 1.0]]], device="cuda")
    y = torch.conv1d(x, k)
    assert y.device.type == "cuda"
    assert y.flatten().tolist() == [3.0, 5.0, 7.0]


def test_imports():
    # magenta_rt installs the import hook for its vendored sequence_layers, so it is imported first
    # (route recorded in requirements-git.txt; the fallback is a direct sequence-layers install).
    import magenta_rt.jax.spectrostream  # noqa: F401
    import sequence_layers.jax  # noqa: F401
    import codicodec  # noqa: F401
    import encodec  # noqa: F401
    from transformers import EncodecModel  # noqa: F401
