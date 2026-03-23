"""Symbolic label formulas for synthetic datasets.

Build labeling rules using the ``F()`` feature proxy and standard Python
operators::

    >>> from concept_benchmark.formula import LabelFormula, F
    >>> formula = LabelFormula(
    ...     score=5 * F("mouth_type").closed + 8 * F("foot_shape").pointy
    ...           - 5 * F("has_knees").true + 2,
    ...     temperature=4.2,
    ... )
    >>> print(formula)
    5·[mouth_type=closed] + 8·[foot_shape=pointy] - 5·[has_knees=true] + 2
    >>> formula.score({"mouth_type": "closed", "foot_shape": "flat", "has_knees": "true"})
    2.0

Interactions are supported via ``&`` (AND), ``|`` (OR), and ``~`` (NOT)::

    >>> spec = LabelFormula(score=3 * (F("a").x & F("b").y))
    >>> spec.score({"a": "x", "b": "y"})
    3.0
    >>> spec.score({"a": "x", "b": "z"})
    0.0
"""

from __future__ import annotations

from dataclasses import dataclass


# ── Expression tree (internal) ───────────────────────────────────────


class _Expr:
    """Base class for expression tree nodes."""

    __slots__ = ()

    def _eval(self, row) -> float:
        raise NotImplementedError

    def _concept_names(self) -> set[str]:
        raise NotImplementedError

    def _to_dict(self) -> dict:
        raise NotImplementedError

    # ── Arithmetic operators ─────────────────────────────────────────

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return _Weighted(float(other), self)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return _Weighted(float(other), self)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = _Const(float(other))
        if isinstance(other, _Expr):
            # Flatten nested sums
            left = self.children if isinstance(self, _Sum) else (self,)
            right = other.children if isinstance(other, _Sum) else (other,)
            return _Sum((*left, *right))
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return _Sum((_Const(float(other)), self))
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self + _Const(-float(other))
        if isinstance(other, _Expr):
            return self + _Weighted(-1.0, other)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return _Sum((_Const(float(other)), _Weighted(-1.0, self)))
        return NotImplemented

    def __neg__(self):
        return _Weighted(-1.0, self)

    # ── Logical operators ────────────────────────────────────────────

    def __and__(self, other):
        if isinstance(other, _Expr):
            return _And(self, other)
        return NotImplemented

    def __or__(self, other):
        if isinstance(other, _Expr):
            return _Or(self, other)
        return NotImplemented

    def __invert__(self):
        return _Not(self)


@dataclass(frozen=True)
class _Indicator(_Expr):
    """1[feature = value]."""

    feature: str
    value: str

    def _eval(self, row) -> float:
        return 1.0 if str(row[self.feature]) == self.value else 0.0

    def _concept_names(self) -> set[str]:
        return {self.feature}

    def _to_dict(self) -> dict:
        return {"type": "indicator", "feature": self.feature, "value": self.value}

    def __str__(self) -> str:
        return f"[{self.feature}={self.value}]"


@dataclass(frozen=True)
class _Const(_Expr):
    """Literal constant."""

    value: float

    def _eval(self, row) -> float:
        return self.value

    def _concept_names(self) -> set[str]:
        return set()

    def _to_dict(self) -> dict:
        return {"type": "const", "value": self.value}

    def __str__(self) -> str:
        return f"{self.value:g}"


@dataclass(frozen=True)
class _Weighted(_Expr):
    """weight × child."""

    weight: float
    child: _Expr

    def _eval(self, row) -> float:
        return self.weight * self.child._eval(row)

    def _concept_names(self) -> set[str]:
        return self.child._concept_names()

    def _to_dict(self) -> dict:
        return {
            "type": "weighted",
            "weight": self.weight,
            "child": self.child._to_dict(),
        }

    def __str__(self) -> str:
        w = self.weight
        child_str = str(self.child)
        if w == 1.0:
            return child_str
        if w == -1.0:
            return f"-{child_str}"
        return f"{w:g}\u00b7{child_str}"


@dataclass(frozen=True)
class _Sum(_Expr):
    """Sum of children."""

    children: tuple[_Expr, ...]

    def _eval(self, row) -> float:
        return sum(c._eval(row) for c in self.children)

    def _concept_names(self) -> set[str]:
        names: set[str] = set()
        for c in self.children:
            names |= c._concept_names()
        return names

    def _to_dict(self) -> dict:
        return {"type": "sum", "children": [c._to_dict() for c in self.children]}

    def __str__(self) -> str:
        parts = []
        for i, c in enumerate(self.children):
            s = str(c)
            if i > 0 and not s.startswith("-"):
                s = "+ " + s
            elif i > 0:
                s = "- " + s[1:]  # strip leading minus, add "- "
            parts.append(s)
        return " ".join(parts)


@dataclass(frozen=True)
class _And(_Expr):
    """Logical AND (product of indicators)."""

    left: _Expr
    right: _Expr

    def _eval(self, row) -> float:
        return 1.0 if self.left._eval(row) and self.right._eval(row) else 0.0

    def _concept_names(self) -> set[str]:
        return self.left._concept_names() | self.right._concept_names()

    def _to_dict(self) -> dict:
        return {
            "type": "and",
            "left": self.left._to_dict(),
            "right": self.right._to_dict(),
        }

    def __str__(self) -> str:
        return f"({self.left} & {self.right})"


@dataclass(frozen=True)
class _Or(_Expr):
    """Logical OR."""

    left: _Expr
    right: _Expr

    def _eval(self, row) -> float:
        return 1.0 if self.left._eval(row) or self.right._eval(row) else 0.0

    def _concept_names(self) -> set[str]:
        return self.left._concept_names() | self.right._concept_names()

    def _to_dict(self) -> dict:
        return {
            "type": "or",
            "left": self.left._to_dict(),
            "right": self.right._to_dict(),
        }

    def __str__(self) -> str:
        return f"({self.left} | {self.right})"


@dataclass(frozen=True)
class _Not(_Expr):
    """Logical NOT."""

    child: _Expr

    def _eval(self, row) -> float:
        return 1.0 - self.child._eval(row)

    def _concept_names(self) -> set[str]:
        return self.child._concept_names()

    def _to_dict(self) -> dict:
        return {"type": "not", "child": self.child._to_dict()}

    def __str__(self) -> str:
        return f"~{self.child}"


# ── Tree deserialization ─────────────────────────────────────────────

_NODE_BUILDERS = {
    "indicator": lambda d: _Indicator(d["feature"], d["value"]),
    "const": lambda d: _Const(d["value"]),
    "weighted": lambda d: _Weighted(d["weight"], _expr_from_dict(d["child"])),
    "sum": lambda d: _Sum(tuple(_expr_from_dict(c) for c in d["children"])),
    "and": lambda d: _And(_expr_from_dict(d["left"]), _expr_from_dict(d["right"])),
    "or": lambda d: _Or(_expr_from_dict(d["left"]), _expr_from_dict(d["right"])),
    "not": lambda d: _Not(_expr_from_dict(d["child"])),
}


def _expr_from_dict(d: dict) -> _Expr:
    builder = _NODE_BUILDERS.get(d.get("type"))
    if builder is None:
        raise ValueError(f"Unknown expression node type: {d.get('type')!r}")
    return builder(d)


# ── Feature proxy ────────────────────────────────────────────────────


class F:
    """Feature proxy for building label formulas.

    ``F("mouth_type").closed`` creates an indicator node ``1[mouth_type=closed]``.
    Use ``.eq(value)`` for values that aren't valid Python identifiers.

    Example::

        >>> mouth = F("mouth_type")
        >>> expr = 5 * mouth.closed + 8 * F("foot_shape").pointy
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, value: str) -> _Indicator:
        if value.startswith("_"):
            raise AttributeError(value)
        return _Indicator(self._name, value)

    def eq(self, value: str) -> _Indicator:
        """Create an indicator for values that aren't valid Python identifiers."""
        return _Indicator(self._name, str(value))

    def __repr__(self) -> str:
        return f"F({self._name!r})"


# ── LabelFormula ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class LabelFormula:
    """Labeling rule for synthetic robot datasets.

    Build formulas using the :class:`F` feature proxy::

        formula = LabelFormula(
            score=5 * F("mouth_type").closed + 8 * F("foot_shape").pointy
                  - 5 * F("has_knees").true + 2,
            temperature=4.2,
            stochastic=True,
        )

    Parameters
    ----------
    score : _Expr
        Expression tree for the linear score.
    temperature : float
        Sigmoid steepness for stochastic labeling (default 1.0).
    stochastic : bool
        If ``True`` (default), ``.label()`` samples via Bernoulli.
        If ``False``, ``.label()`` uses deterministic threshold (score ≥ 0 → Glorp).
    """

    _score_expr: _Expr
    temperature: float = 1.0
    stochastic: bool = True

    def __init__(
        self,
        *,
        score: _Expr,
        temperature: float = 1.0,
        stochastic: bool = True,
    ) -> None:
        if isinstance(score, (int, float)):
            score = _Const(float(score))
        if not isinstance(score, _Expr):
            raise TypeError(
                f"score must be an expression built with F(), got {type(score).__name__}"
            )
        object.__setattr__(self, "_score_expr", score)
        object.__setattr__(self, "temperature", float(temperature))
        object.__setattr__(self, "stochastic", bool(stochastic))

    # ── Computation ──────────────────────────────────────────────────

    def score(self, row) -> float:
        """Evaluate the score expression on a single sample (dict or Series)."""
        return self._score_expr._eval(row)

    def probability(self, row) -> float:
        """Compute P(Glorp) = σ(temperature × score)."""
        from scipy.special import expit

        return float(expit(self.temperature * self.score(row)))

    def label(self, row, rng=None) -> str:
        """Return ``"glorp"`` or ``"drent"``.

        If the formula is stochastic, samples from P(Glorp) using *rng*
        (or a default RNG if *rng* is ``None``).
        If deterministic, returns Glorp when score ≥ 0.
        """
        if self.stochastic:
            if rng is None:
                import numpy as np

                rng = np.random.default_rng()
            return "glorp" if rng.random() < self.probability(row) else "drent"
        return "glorp" if self.score(row) >= 0 else "drent"

    # ── Properties ───────────────────────────────────────────────────

    @property
    def concept_names(self) -> set[str]:
        """Set of all concept/feature names referenced in the formula."""
        return self._score_expr._concept_names()

    @property
    def features(self) -> dict[str, str]:
        """Feature → value mapping (only for linear formulas)."""
        terms = self._extract_linear_terms()
        return {f: v for f, v, _ in terms}

    @property
    def weights(self) -> dict[str, float]:
        """Feature → weight mapping (only for linear formulas)."""
        terms = self._extract_linear_terms()
        return {f: w for f, _, w in terms}

    @property
    def intercept(self) -> float:
        """Intercept term (only for linear formulas)."""
        _, intercept = self._decompose_linear()
        return intercept

    def _extract_linear_terms(self) -> list[tuple[str, str, float]]:
        """Extract (feature, value, weight) tuples from a linear formula."""
        terms, _ = self._decompose_linear()
        return terms

    def _decompose_linear(self) -> tuple[list[tuple[str, str, float]], float]:
        """Decompose into linear terms + intercept. Raises if non-linear."""
        terms = []
        intercept = 0.0
        nodes = (
            list(self._score_expr.children) if isinstance(self._score_expr, _Sum) else [self._score_expr]
        )
        for node in nodes:
            if isinstance(node, _Const):
                intercept += node.value
            elif isinstance(node, _Indicator):
                terms.append((node.feature, node.value, 1.0))
            elif isinstance(node, _Weighted):
                # Unwrap nested weights: w1 * (w2 * indicator)
                w, inner = node.weight, node.child
                while isinstance(inner, _Weighted):
                    w *= inner.weight
                    inner = inner.child
                if isinstance(inner, _Indicator):
                    terms.append((inner.feature, inner.value, w))
                else:
                    raise TypeError(f"Cannot decompose non-linear expression: {node}")
            else:
                raise TypeError(f"Cannot decompose non-linear expression: {node}")
        return terms, intercept

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to a dict for YAML/JSON storage."""
        return {
            "score": self._score_expr._to_dict(),
            "temperature": self.temperature,
            "stochastic": self.stochastic,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LabelFormula:
        """Construct from a serialized dict.

        Accepts both the new tree format (``{"score": {...}, ...}``)
        and the legacy flat format (``{"terms": {...}, "intercept": ..., ...}``).
        """
        if "score" in d:
            return cls(
                score=_expr_from_dict(d["score"]),
                temperature=d.get("temperature", 1.0),
                stochastic=d.get("stochastic", True),
            )

        # Legacy flat format
        if "terms" not in d:
            raise ValueError('LabelFormula dict must have "score" or "terms" key.')
        for f, spec in d["terms"].items():
            if not isinstance(spec, dict) or "value" not in spec or "weight" not in spec:
                raise ValueError(
                    f"Each term must be a dict with 'value' and 'weight' keys, "
                    f"got {spec!r} for feature {f!r}"
                )
        return cls.from_weights(
            features={f: spec["value"] for f, spec in d["terms"].items()},
            weights={f: spec["weight"] for f, spec in d["terms"].items()},
            intercept=d.get("intercept", 0.0),
            temperature=d.get("temperature", 1.0),
        )

    @classmethod
    def from_weights(
        cls,
        features: dict[str, str],
        weights: dict[str, float] | None = None,
        intercept: float = 0.0,
        temperature: float = 1.0,
        stochastic: bool = True,
    ) -> LabelFormula:
        """Construct from flat feature/weight dicts (bridge for legacy code).

        Parameters
        ----------
        features : dict
            Feature name → value mapping.
        weights : dict, optional
            Feature name → weight mapping. Defaults to 1.0 for all.
        intercept : float
            Bias term added to the score.
        temperature : float
            Sigmoid steepness.
        stochastic : bool
            Whether to sample labels stochastically.
        """
        weights = weights or {}
        terms: list[_Expr] = []
        for feat, val in features.items():
            w = float(weights.get(feat, 1.0))
            ind = _Indicator(feat, str(val))
            terms.append(_Weighted(w, ind) if w != 1.0 else ind)
        if intercept:
            terms.append(_Const(float(intercept)))
        score_expr = (
            _Sum(tuple(terms))
            if len(terms) > 1
            else (terms[0] if terms else _Const(0.0))
        )
        return cls(score=score_expr, temperature=temperature, stochastic=stochastic)

    # ── Validation ───────────────────────────────────────────────────

    def validate_against(self, concepts: dict) -> None:
        """Check that all formula features and values exist in *concepts*."""
        for node in self._iter_indicators():
            if node.feature not in concepts:
                raise ValueError(
                    f"Formula feature {node.feature!r} not found in concepts. "
                    f"Available: {sorted(concepts)}"
                )
            valid = [str(v) for v in concepts[node.feature]]
            if node.value not in valid:
                if not any(
                    node.value.startswith(str(v)) or str(v).startswith(node.value)
                    for v in concepts[node.feature]
                ):
                    raise ValueError(
                        f"Value {node.value!r} not valid for feature {node.feature!r}. "
                        f"Valid values: {valid}"
                    )

    def _iter_indicators(self):
        """Yield all _Indicator nodes in the tree."""
        stack = [self._score_expr]
        while stack:
            node = stack.pop()
            if isinstance(node, _Indicator):
                yield node
            elif isinstance(node, _Weighted):
                stack.append(node.child)
            elif isinstance(node, _Sum):
                stack.extend(node.children)
            elif isinstance(node, (_And, _Or)):
                stack.extend([node.left, node.right])
            elif isinstance(node, _Not):
                stack.append(node.child)

    # ── Display ──────────────────────────────────────────────────────

    def __str__(self) -> str:
        return str(self._score_expr)

    def __repr__(self) -> str:
        mode = "stochastic" if self.stochastic else "deterministic"
        return (
            f"LabelFormula({mode}, temperature={self.temperature})\n"
            f"  {self._score_expr}"
        )
