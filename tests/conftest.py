import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed() -> None:
    np.random.seed(0)


@pytest.fixture(autouse=True)
def _torch_dtype(request: pytest.FixtureRequest):  # noqa: ANN201
    """The merged thermal tests set float64 as torch's default when imported; every other test expects float32."""
    try:
        import torch
    except ImportError:
        yield
        return
    want = torch.float64 if "thermal" in str(request.node.fspath) else torch.float32
    before = torch.get_default_dtype()
    torch.set_default_dtype(want)
    yield
    torch.set_default_dtype(before)
