from __future__ import annotations

import pytest

pytest.importorskip("lm_eval")

torch = pytest.importorskip("torch")

from arc.lmeval import ArcLM, ArcCausalLMShim  # noqa: E402

TINY_VOCAB = 128


def tiny_word_tokenizer():
    """Real fast tokenizer whose vocab fits the tiny model's 128 embeddings."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = [str(i) for i in range(10)] + [
        "+", "-", "*", "/", "=", "The", "question", "answer", "is",
    ]
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )


@pytest.fixture(scope="module")
def tokenizer():
    return tiny_word_tokenizer()


@pytest.fixture(scope="module")
def tiny_arc_lm(tokenizer):
    return ArcLM(
        scale="model",
        recurrence=2,
        path="tiny",
        tokenizer=tokenizer,
        batch_size=1,
    )


def test_arc_lm_builds_all_scales(tokenizer):
    for scale in ("base", "layer", "block", "model", "l", "b", "m"):
        lm = ArcLM(scale=scale, recurrence=1, path="tiny", tokenizer=tokenizer)
        assert isinstance(lm.arc_model, ArcCausalLMShim)
        assert lm.arc_model.config.vocab_size == TINY_VOCAB
    with pytest.raises(ValueError, match="unknown scale"):
        ArcLM(scale="nope", path="tiny", tokenizer=tokenizer)


def test_loglikelihood_through_harness(tiny_arc_lm):
    from lm_eval.api.instance import Instance

    pairs = [("81 / 9 =", "9"), ("2 + 2 =", "4")]
    requests = [Instance("loglikelihood", doc={}, arguments=p, idx=0) for p in pairs]
    results = tiny_arc_lm.loglikelihood(requests)

    assert len(results) == 2
    for ll, greedy in results:
        assert isinstance(ll, float) and ll < 0.0
        assert isinstance(greedy, bool)
    shim = tiny_arc_lm.arc_model
    assert shim.forward_count == 2
    assert shim.total_flops_used > 0
    assert shim.total_executions == 4  # model recurrence x2 -> 2 executions/request


def test_compute_accounting_respects_recurrence(tokenizer):
    base_lm = ArcLM(scale="base", path="tiny", tokenizer=tokenizer)
    ids = tokenizer("81 / 9 = 9", return_tensors="pt").input_ids

    base_lm.arc_model(ids)
    base_flops = base_lm.arc_model.total_flops_used
    assert base_lm.arc_model.total_executions == 1

    tiny_arc_lm = ArcLM(scale="model", recurrence=2, path="tiny", tokenizer=tokenizer)
    tiny_arc_lm.arc_model(ids)
    recurrent_flops = tiny_arc_lm.arc_model.total_flops_used

    assert recurrent_flops == pytest.approx(2 * base_flops, rel=0.05)


def test_shim_greedy_generate(tiny_arc_lm):
    shim = tiny_arc_lm.arc_model
    ids = tiny_arc_lm.tokenizer("2 + 2 =", return_tensors="pt").input_ids

    out = shim.generate(input_ids=ids, max_length=ids.shape[1] + 3)
    assert out.shape == (1, ids.shape[1] + 3)

    # greedy decoding is deterministic
    again = shim.generate(input_ids=ids, max_length=ids.shape[1] + 3)
    assert torch.equal(out, again)

    # accounting flows through the same forward path
    flops_before = shim.total_flops_used
    shim.generate(input_ids=ids, max_length=ids.shape[1] + 1)
    assert shim.total_flops_used > flops_before


def test_shim_generate_rejects_sampling(tiny_arc_lm):
    ids = tiny_arc_lm.tokenizer("2 + 2 =", return_tensors="pt").input_ids
    with pytest.raises(NotImplementedError, match="greedy-only"):
        tiny_arc_lm.arc_model.generate(input_ids=ids, max_length=ids.shape[1] + 1, do_sample=True)


def test_simple_evaluate_end_to_end(tiny_arc_lm):
    """Full harness round-trip on a real task. Skips without internet."""
    try:
        from lm_eval import simple_evaluate

        results = simple_evaluate(
            model=tiny_arc_lm,
            tasks=["arc_easy"],
            limit=2,
            random_seed=0,
            numpy_random_seed=0,
            torch_random_seed=0,
            fewshot_random_seed=0,
        )
    except Exception as e:  # noqa: BLE001 - network errors vary by datasets version
        name = type(e).__name__.lower()
        msg = str(e).lower()
        if any(k in name + msg for k in ("connection", "offline", "timeout", "http", "url")):
            pytest.skip(f"offline: {e}")
        raise

    arc_easy = results["results"]["arc_easy"]
    assert "acc,none" in arc_easy
    assert tiny_arc_lm.arc_model.total_flops_used > 0
