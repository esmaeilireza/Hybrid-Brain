"""Simulation time loop - single source of time truth in the entire brain.

Rule: no module counts time itself; everyone gets t from here.
"""
from dataclasses import dataclass, field


@dataclass
class BrainClock:
    """Millisecond simulation clock."""

    dt_ms: float = 1.0
    max_time_ms: float = 60_000.0

    _t_ms: float = field(default=0.0, init=False, repr=False)
    _tick: int = field(default=0, init=False, repr=False)

    @property
    def t_ms(self) -> float:
        return self._t_ms

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def done(self) -> bool:
        return self._t_ms >= self.max_time_ms

    def advance(self) -> float:
        """Advance one tick - returns t (time after the tick)."""
        if self.done:
            raise RuntimeError("Simulation has reached the time limit")
        self._tick += 1
        self._t_ms += self.dt_ms
        return self._t_ms

    def reset(self) -> None:
        self._t_ms = 0.0
        self._tick = 0
