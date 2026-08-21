from __future__ import annotations

import pytest

from arc.models.jetmoe import JetMoeAdapter, build_tiny_jetmoe

TINY_MODEL_PATH = "models/jetmoe-8b"


@pytest.fixture(scope="session")
def tiny_model():
    return build_tiny_jetmoe(seed=0)


@pytest.fixture(scope="session")
def layer_adapter(tiny_model):
    return JetMoeAdapter(tiny_model, block_size=1)


@pytest.fixture(scope="session")
def block_adapter(tiny_model):
    return JetMoeAdapter(tiny_model, block_size=2)


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TINY_MODEL_PATH)
