import numpy as np
import pytest
import torch
from pathlib import Path
from PIL import Image

from concept_benchmark.data import ConceptImageDatasetSample


def test_len_and_repr(img_small):
    s = img_small.training
    assert isinstance(s, ConceptImageDatasetSample)
    assert len(s) == s.n
    repr(s)  # smoke


def test_getitem_returns_pil_and_int_tensors(img_small):
    s = img_small.training
    image, c, y = s[0]
    assert isinstance(image, Image.Image)
    assert isinstance(c, torch.Tensor) and c.dtype == torch.int64 and c.ndim == 1
    assert isinstance(y, torch.Tensor) and y.dtype == torch.int64 and y.ndim == 0


def test_transform_x_to_tensor_and_loader_batching(img_small):
    # Convert PIL->CHW float32 tensor
    def to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:  # grayscale -> [1,H,W]
            arr = arr[None, ...]
        else:              # HWC->CHW
            arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr)

    s = ConceptImageDatasetSample(
        parent=img_small,
        X=img_small.X,
        C=img_small.C,
        y=img_small.y,
        meta=img_small._full.meta,
        base_dir=img_small._full.base_dir,
        preprocess=None,
        transform_x=to_tensor,
    )

    x, c, y = s[0]
    assert isinstance(x, torch.Tensor) and x.dtype == torch.float32 and x.ndim == 3

    loader = s.loader(batch_size=4, shuffle=False, num_workers=0)
    xb, cb, yb = next(iter(loader))
    assert isinstance(xb, torch.Tensor) and xb.ndim == 4 and xb.shape[0] == 4
    assert isinstance(cb, torch.Tensor) and cb.dtype == torch.int64 and cb.ndim == 2
    assert isinstance(yb, torch.Tensor) and yb.dtype == torch.int64 and yb.ndim == 1


def test_missing_image_warns_and_returns_path(img_with_missing):
    s = img_with_missing.training
    # In our factory, the missing path is appended at the end
    idx_missing = len(s) - 1
    with pytest.warns(RuntimeWarning, match="cannot open image, returning path"):
        image, c, y = s[idx_missing]
    from pathlib import Path as _P
    assert isinstance(image, (str, _P))
    assert isinstance(c, torch.Tensor) and isinstance(y, torch.Tensor)


def test_equality_checks_base_dir_and_preprocess(img_small, tmp_path):
    base1 = tmp_path / "a"
    base2 = tmp_path / "b"
    s1 = ConceptImageDatasetSample(
        parent=img_small, X=img_small.X, C=img_small.C, y=img_small.y,
        meta=img_small._full.meta, base_dir=base1,
    )
    s2 = ConceptImageDatasetSample(
        parent=img_small, X=img_small.X, C=img_small.C, y=img_small.y,
        meta=img_small._full.meta, base_dir=base2,
    )
    assert s1 != s2  # base_dir differs

    def pp(img): return img
    s3 = ConceptImageDatasetSample(
        parent=img_small, X=img_small.X, C=img_small.C, y=img_small.y,
        meta=img_small._full.meta, base_dir=base1, preprocess=pp,
    )
    assert s1 != s3  # preprocess differs