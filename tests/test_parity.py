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


def test_model_recurrence_chains_normalized_state(layer_adapter):
    import torch.nn.functional as F

    ids = torch.randint(0, 128, (1, 8))
    model = __import__("arc.recurrence", fromlist=["build_recurrent_model"]).build_recurrent_model(
        "model", layer_adapter, recurrence=2
    )

    with torch.no_grad():
        hidden = layer_adapter.embed(ids)
        ctx = layer_adapter.prepare(hidden)
        hidden = layer_adapter.forward_model(hidden, ctx)
        hidden = layer_adapter.normalize(hidden)
        hidden = layer_adapter.forward_model(hidden, ctx)
        expected = layer_adapter.final_logits(hidden)
        actual = model(ids).logits

    assert torch.allclose(actual, expected, atol=1e-5)


def test_decomposed_forward_matches_native_with_padding(layer_adapter, tiny_model):
    ids = torch.randint(2, 128, (2, 8))
    ids[0, :3] = 0
    attention_mask = torch.ones_like(ids)
    attention_mask[0, :3] = 0
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)

    with torch.no_grad():
        native = tiny_model(
            ids, attention_mask=attention_mask, position_ids=position_ids
        ).logits
        hidden = layer_adapter.embed(ids)
        ctx = layer_adapter.prepare(hidden, attention_mask, position_ids)
        hidden = layer_adapter.forward_model(hidden, ctx)
        decomposed = layer_adapter.final_logits(hidden)

    assert torch.allclose(native, decomposed, atol=1e-5)


def test_result_final_hidden_is_normalized(layer_adapter):
    from arc.recurrence import BaseLM, build_recurrent_model

    ids = torch.randint(0, 128, (1, 8))
    base = BaseLM(layer_adapter)(ids)
    recurrent = build_recurrent_model("layer", layer_adapter, recurrence=2)(ids)

    assert torch.allclose(base.logits, layer_adapter.project_logits(base.final_hidden))
    assert torch.allclose(
        recurrent.logits, layer_adapter.project_logits(recurrent.final_hidden)
    )


def test_model_recurrence_uses_previous_loop_state(layer_adapter):
    ids = torch.randint(0, 128, (1, 8))
    model = __import__("arc.recurrence", fromlist=["build_recurrent_model"]).build_recurrent_model(
        "model", layer_adapter, recurrence=2
    )
    observed_inputs = []
    original = layer_adapter.forward_model

    def record_input(hidden, ctx):
        observed_inputs.append(hidden.detach().clone())
        return original(hidden, ctx)

    layer_adapter.forward_model = record_input
    try:
        result = model(ids)
    finally:
        layer_adapter.forward_model = original

    assert len(observed_inputs) == 2
    assert not torch.allclose(observed_inputs[0], observed_inputs[1])
    first_output = original(observed_inputs[0], layer_adapter.prepare(observed_inputs[0]))
    assert torch.allclose(observed_inputs[1], layer_adapter.normalize(first_output), atol=1e-5)
    assert result.state.executions == 2
