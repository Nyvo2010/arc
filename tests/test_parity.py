from __future__ import annotations

import torch


def test_decomposed_forward_matches_native(layer_adapter, tiny_model):
    torch.manual_seed(1)
    ids = torch.randint(0, 128, (1, 16))

    with torch.no_grad():
        native_logits = tiny_model(ids).logits
        h = layer_adapter.embed(ids)
        ctx = layer_adapter.prepare(h)
        h = layer_adapter.forward_model(h, ctx)
        decomposed_logits = layer_adapter.final_logits(h)

    max_diff = (native_logits - decomposed_logits).abs().max().item()
    assert max_diff < 1e-5, f"adapter parity broken: max diff {max_diff}"


def test_block_partition_matches_native(block_adapter, tiny_model):
    torch.manual_seed(2)
    ids = torch.randint(0, 128, (2, 8))
    num_blocks = block_adapter.num_blocks()

    with torch.no_grad():
        native_logits = tiny_model(ids).logits
        h = block_adapter.embed(ids)
        ctx = block_adapter.prepare(h)
        for b in range(num_blocks):
            h = block_adapter.forward_block(b, h, ctx)
        logits = block_adapter.final_logits(h)

    assert (native_logits - logits).abs().max().item() < 1e-5


def test_layer_and_block_paths_agree(layer_adapter, block_adapter):
    torch.manual_seed(3)
    ids = torch.randint(0, 128, (1, 12))

    with torch.no_grad():
        ha = layer_adapter.embed(ids)
        ca = layer_adapter.prepare(ha)
        for i in range(layer_adapter.num_layers()):
            ha = layer_adapter.forward_layer(i, ha, ca)

        hb = block_adapter.embed(ids)
        cb = block_adapter.prepare(hb)
        for b in range(block_adapter.num_blocks()):
            hb = block_adapter.forward_block(b, hb, cb)

    assert torch.allclose(ha, hb, atol=1e-6)
