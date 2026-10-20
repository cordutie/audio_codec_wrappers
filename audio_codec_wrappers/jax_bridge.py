"""Zero-copy GPU transfers between torch and jax (DLPack), design D16."""
import torch

from . import configure_jax_env

# make sure preallocation is off before jax is imported below
configure_jax_env()

import jax  # noqa: E402


def torch_to_jax(t: torch.Tensor) -> "jax.Array":
    """Torch tensor to jax array through DLPack (zero copy on GPU)."""
    return jax.dlpack.from_dlpack(t.detach().contiguous())


def jax_to_torch(a: "jax.Array") -> torch.Tensor:
    """Jax array to torch tensor through DLPack (zero copy on GPU)."""
    return torch.from_dlpack(a)
