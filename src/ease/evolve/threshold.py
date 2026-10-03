"""A threshold that tracks the observed error rate, with a bound that needs no assumptions.

The assistant endorses an action only when its confidence reaches a threshold tau. After an
endorsed action a person says whether it was right, and the threshold moves:

    tau <- tau + eta * (e - alpha)        e = 1 if that endorsement was wrong, else 0

An error raises tau by eta * (1 - alpha); a correct endorsement lowers it by eta * alpha. This is
the quantile-tracking form of online conformal prediction (Gibbs & Candes, 2021; Angelopoulos,
Candes & Tibshirani, 2023), applied to endorsements instead of prediction sets.

Guarantee (docs/PROOFS.md, T5; tests/test_evolve.py checks it on adversarial sequences)

    Let D be the largest number of endorsements ever awaiting a verdict at the same time
    (D = 1 when every verdict arrives before the next endorsement). After K verdicts,

        (1/K) * sum_k e_k  <=  alpha + (1 + D * eta * (1 - alpha) - tau_1) / (eta * K).

    Reason: summing the update gives  sum_k (e_k - alpha) = (tau_final - tau_1) / eta.  Confidence
    never exceeds 1, so tau <= 1 whenever something is endorsed; after the last endorsement at
    most D verdicts remain, each raising tau by at most eta * (1 - alpha).

    This holds for every sequence of cases, adversarial ones included, and under any shift in the
    data. It assumes only that verdicts are truthful.

What the bound does not say
    It is a long-run frequency over endorsed actions that received a verdict. It is not a
    probability for an individual decision. It can be met by endorsing very little: if the model
    cannot tell right from wrong, tau climbs above 1 and the assistant stops endorsing. `stalled`
    reports that state. The remedy is a better model, after which a new tracker starts and its
    bound counts from zero.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ThresholdTracker:
    alpha: float = 0.05
    eta: float = 0.05
    tau: float = 0.9
    tau_initial: float = 0.9
    verdicts: int = 0
    errors: int = 0
    outstanding: int = 0
    max_outstanding: int = 1

    def __post_init__(self):
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")
        if self.eta <= 0:
            raise ValueError("eta must be positive")

    def endorses(self, confidence: float) -> bool:
        return confidence >= self.tau

    def endorse(self, confidence: float) -> bool:
        """Decide, and if the answer is yes, register that a verdict is now awaited."""
        if confidence > 1.0 + 1e-12:
            raise ValueError("confidence cannot exceed 1")
        ok = self.endorses(confidence)
        if ok:
            self.outstanding += 1
            self.max_outstanding = max(self.max_outstanding, self.outstanding)
        return ok

    def verdict(self, was_wrong: bool) -> float:
        """Record the verdict on an endorsed action and return the new threshold."""
        if self.outstanding <= 0:
            raise RuntimeError("a verdict arrived for which no endorsement is outstanding")
        self.outstanding -= 1
        self.verdicts += 1
        self.errors += int(bool(was_wrong))
        self.tau += self.eta * ((1.0 if was_wrong else 0.0) - self.alpha)
        return self.tau

    def withdraw(self) -> None:
        """An endorsement will never receive a verdict (the task was abandoned)."""
        if self.outstanding > 0:
            self.outstanding -= 1

    @property
    def error_rate(self) -> float:
        return self.errors / self.verdicts if self.verdicts else 0.0

    @property
    def bound(self) -> float:
        if self.verdicts == 0:
            return 1.0
        slack = 1.0 + self.max_outstanding * self.eta * (1.0 - self.alpha) - self.tau_initial
        return min(1.0, self.alpha + slack / (self.eta * self.verdicts))

    @property
    def stalled(self) -> bool:
        return self.tau > 1.0

    def state(self) -> dict:
        return {"alpha": self.alpha, "eta": self.eta, "tau": self.tau, "tau_initial": self.tau_initial,
                "verdicts": self.verdicts, "errors": self.errors, "outstanding": self.outstanding,
                "max_outstanding": self.max_outstanding, "error_rate": self.error_rate, "bound": self.bound,
                "stalled": self.stalled}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ThresholdTracker":
        d = json.loads(Path(path).read_text())
        return cls(**{k: d[k] for k in ("alpha", "eta", "tau", "tau_initial", "verdicts", "errors", "outstanding",
                                        "max_outstanding")})
