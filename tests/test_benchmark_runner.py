from arc.benchmarks.protocol import BenchmarkProtocol, ModelVariant
from arc.benchmarks.runner import run_suite


def test_runner_freezes_protocol_and_records_accounting(monkeypatch):
    import arc.lmeval

    class Shim:
        total_flops_used = 12.5
        total_executions = 3

    class Model:
        arc_model = Shim()

    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs)
        arc.lmeval.ArcLM.last_instance = Model()
        return {"results": {"mmlu": {"acc,none": 0.5}}}

    monkeypatch.setattr("lm_eval.simple_evaluate", evaluate)
    variants = [
        ModelVariant("jetmoe", "a", scale="l", recurrence=2),
        ModelVariant("jetmoe", "b", scale="m", recurrence=3),
    ]
    protocol = BenchmarkProtocol(limit=2)
    output = run_suite(variants, protocol)

    assert len(calls) == 2
    assert calls[0]["tasks"] == calls[1]["tasks"] == list(protocol.tasks)
    assert calls[0]["random_seed"] == calls[1]["random_seed"] == 1234
    assert output["runs"][0]["accounting"] == {
        "total_flops_used": 12.5,
        "total_executions": 3,
    }
