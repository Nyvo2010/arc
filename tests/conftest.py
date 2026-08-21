from __future__ import annotations

import pytest

from arc.models.jetmoe import JetMoeAdapter, build_tiny_jetmoe


@pytest.fixture(scope="session")
def tiny_model():
    return build_tiny_jetmoe(seed=0)


@pytest.fixture(scope="session")
def layer_adapter(tiny_model):
    return JetMoeAdapter(tiny_model, block_size=1)


@pytest.fixture(scope="session")
def block_adapter(tiny_model):
    return JetMoeAdapter(tiny_model, block_size=2)
