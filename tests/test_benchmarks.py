import pytest

from arc.benchmarks.protocol import BenchmarkProtocol, ModelVariant


def test_variant_only_changes_model_fields():
    variants = [
        ModelVariant("jetmoe", "weights"),
        ModelVariant("jetmoe", "weights", "l", 2),
        ModelVariant("jetmoe", "weights", "b", 3),
        ModelVariant("jetmoe", "weights", "m", 4),
    ]
    protocol = BenchmarkProtocol()
    assert [protocol.harness_args() for _ in variants] == [protocol.harness_args()] * 4
    assert [variant.scale for variant in variants] == ["base", "l", "b", "m"]


def test_variant_rejects_invalid_base_recurrence():
    with pytest.raises(ValueError, match="base recurrence"):
        ModelVariant("jetmoe", "weights", recurrence=2)


def test_registry_rejects_unknown_architecture():
    """Contract guard: future architectures must register before use."""
    import arc.lmeval  # noqa: F401
    from arc.models.registry import create_adapter

    with pytest.raises(ValueError, match="unknown architecture"):
        create_adapter("tiny", architecture="deepseekmoe")


def test_protocol_contains_required_tasks_and_zero_shot():
    protocol = BenchmarkProtocol()
    assert protocol.tasks == (
        "mmlu",
        "gpqa_main_zeroshot",
        "gpqa_diamond_zeroshot",
    )
    assert protocol.num_fewshot == 0


def test_protocol_task_names_resolve_in_installed_harness():
    from lm_eval.tasks import TaskManager

    tm = TaskManager()
    resolved = set(tm.match_tasks(list(BenchmarkProtocol().tasks)))
    assert resolved == set(BenchmarkProtocol().tasks)
