from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FixedSchedule:
    counts: dict[int, int]

    @classmethod
    def uniform(cls, num_units: int, count: int) -> "FixedSchedule":
        return cls({i: count for i in range(num_units)})

    @classmethod
    def heterogeneous(cls, counts: list[int]) -> "FixedSchedule":
        return cls({i: c for i, c in enumerate(counts)})

    def count_for(self, unit_index: int) -> int:
        return self.counts.get(unit_index, 1)

    @property
    def total_executions(self) -> int:
        return sum(self.counts.values())
