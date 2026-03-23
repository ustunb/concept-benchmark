"""Label formula for synthetic robot datasets.

A :class:`LabelFormula` encodes the rule that maps robot features to
Glorp/Drent labels.  It computes:

    score = Σ wᵢ · 1[featureᵢ = valueᵢ] + intercept

Then either:
- **Deterministic:** Glorp if score ≥ 0, else Drent.
- **Stochastic:** P(Glorp) = σ(temperature × score), sampled via Bernoulli.

Example::

    >>> formula = LabelFormula(
    ...     mouth_type=("closed", 5.0),
    ...     foot_shape=("pointy", 8.0),
    ...     has_knees=("true", -5.0),
    ...     intercept=2.0,
    ...     temperature=4.2,
    ... )
    >>> print(formula)
    +5.0·[mouth_type=closed] +8.0·[foot_shape=pointy] -5.0·[has_knees=true] +2.0
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelFormula:
    """Labeling rule for synthetic robot datasets.

    Parameters are passed as keyword arguments where each feature is a
    ``(value, weight)`` tuple.  ``intercept`` and ``temperature`` are
    reserved names for the bias term and sigmoid steepness.

    Parameters
    ----------
    intercept : float
        Constant added to the score (default 0.0).
    temperature : float
        Sigmoid steepness for stochastic labeling (default 1.0).
    **features
        Keyword arguments of the form ``feature_name=(value, weight)``.
    """

    terms: tuple[tuple[str, str, float], ...]
    intercept: float = 0.0
    temperature: float = 1.0

    def __init__(
        self,
        *,
        intercept: float = 0.0,
        temperature: float = 1.0,
        **features: tuple[str, float],
    ) -> None:
        terms = tuple(
            (feat, str(val), float(weight)) for feat, (val, weight) in features.items()
        )
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "intercept", float(intercept))
        object.__setattr__(self, "temperature", float(temperature))

    # ── Computation ──────────────────────────────────────────────────

    def score(self, row) -> float:
        """Compute the linear score for a single sample (dict or Series)."""
        return sum(w * int(row[f] == v) for f, v, w in self.terms) + self.intercept

    def probability(self, row) -> float:
        """Compute P(Glorp) = σ(temperature × score)."""
        from scipy.special import expit

        return float(expit(self.temperature * self.score(row)))

    def label(self, row, rng=None) -> str:
        """Return ``"glorp"`` or ``"drent"``.

        If *rng* is provided, sample stochastically from the probability.
        Otherwise, return the deterministic label (score ≥ 0 → Glorp).
        """
        if rng is not None:
            return "glorp" if rng.random() < self.probability(row) else "drent"
        return "glorp" if self.score(row) >= 0 else "drent"

    # ── Properties ───────────────────────────────────────────────────

    @property
    def features(self) -> dict[str, str]:
        """Feature → value mapping."""
        return {f: v for f, v, _ in self.terms}

    @property
    def weights(self) -> dict[str, float]:
        """Feature → weight mapping."""
        return {f: w for f, _, w in self.terms}

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Convert to the nested dict format used in YAML configs."""
        return {
            "terms": {f: {"value": v, "weight": w} for f, v, w in self.terms},
            "intercept": self.intercept,
            "temperature": self.temperature,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LabelFormula:
        """Construct from a nested dict (as stored in YAML)."""
        if "terms" not in d:
            raise ValueError(
                'LabelFormula dict must have a "terms" key. Expected format:\n'
                '  {"terms": {"feature": {"value": "v", "weight": 1.0}}, '
                '"intercept": 0.0, "temperature": 1.0}'
            )
        for f, spec in d["terms"].items():
            if (
                not isinstance(spec, dict)
                or "value" not in spec
                or "weight" not in spec
            ):
                raise ValueError(
                    f"Each term must be a dict with 'value' and 'weight' keys, "
                    f"got {spec!r} for feature {f!r}"
                )
        kw = {f: (spec["value"], spec["weight"]) for f, spec in d["terms"].items()}
        return cls(
            intercept=d.get("intercept", 0.0),
            temperature=d.get("temperature", 1.0),
            **kw,
        )

    # ── Validation ───────────────────────────────────────────────────

    def validate_against(self, concepts: dict) -> None:
        """Check that all formula features exist in *concepts*.

        Raises :class:`ValueError` if a feature name or value is invalid.
        Allows prefix matching (e.g. ``"pointy"`` matches a concept whose
        values include ``"pointy"``).
        """
        for feat, val, _ in self.terms:
            if feat not in concepts:
                raise ValueError(
                    f"Formula feature {feat!r} not found in concepts. "
                    f"Available: {sorted(concepts)}"
                )
            valid = [str(v) for v in concepts[feat]]
            if val not in valid:
                if not any(
                    val.startswith(str(v)) or str(v).startswith(val)
                    for v in concepts[feat]
                ):
                    raise ValueError(
                        f"Value {val!r} not valid for feature {feat!r}. "
                        f"Valid values: {valid}"
                    )

    # ── Display ──────────────────────────────────────────────────────

    def __str__(self) -> str:
        parts = [f"{w:+.1f}\u00b7[{f}={v}]" for f, v, w in self.terms]
        if self.intercept:
            parts.append(f"{self.intercept:+.1f}")
        return " ".join(parts)

    def __repr__(self) -> str:
        args = ", ".join(f'{f}=("{v}", {w})' for f, v, w in self.terms)
        extras = f", intercept={self.intercept}, temperature={self.temperature}"
        return f"LabelFormula({args}{extras})"
