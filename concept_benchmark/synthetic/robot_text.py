from __future__ import annotations

import numpy as np
from typing import Sequence, Optional, Any, Callable

from concept_benchmark.data import ConceptDataset, ConceptDatasetSample


class TextConceptDataset(ConceptDataset):
    def __init__(
        self,
        X: Sequence[str],
        C: np.ndarray,
        y: np.ndarray,
        meta: dict,
        cvindices: dict | None = None,
        transform: Callable | None = None,
        concept_transform: Callable | None = None,
        target_transform: Callable | None = None,
        **kwargs: Any,
    ) -> None:
        meta = dict(meta or {})
        meta.setdefault("data_type", "text")
        if "classes" not in meta:
            meta["classes"] = list(sorted(map(int, np.unique(y))))
        if "concepts" not in meta:
            meta["concepts"] = [f"c{i}" for i in range(np.asarray(C).shape[1])]

        X_arr = np.asarray(list(X), dtype=object)
        C_arr = np.asarray(C, dtype=np.int8)
        y_arr = np.asarray(y, dtype=np.int32)

        self._full = ConceptDatasetSample(
            parent=self,
            X=X_arr,
            C=C_arr,
            y=y_arr,
            meta=meta,
            transform=transform,
            concept_transform=concept_transform,
            target_transform=target_transform,
            **kwargs,
        )
        self.cvindices = {} if cvindices is None else cvindices
        self.reset()
        assert self.__check_rep__()

