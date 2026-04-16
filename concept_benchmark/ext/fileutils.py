from __future__ import annotations

import io
import logging
from pathlib import Path
import dill
import pandas as pd

logger = logging.getLogger(__name__)


def save(obj, path, msg=True, overwrite=False, check_save=False, mkdir=True):
    """
    saves data as a pickle file on disk
    :param obj: object to save to disk
    :param path: path to create
    :return: saved path
    """
    f = Path(path)
    if f.is_file() and overwrite is False:
        raise OSError(f"file: {f} exists")

    if not f.parent.exists() and mkdir:
        f.parent.mkdir(parents=True, exist_ok=True)

    with open(f, "wb") as outfile:
        dill.dump({"data": obj}, outfile, protocol=dill.HIGHEST_PROTOCOL)

    if check_save:
        loaded_obj = load(f)
        if isinstance(loaded_obj, pd.DataFrame):
            if not loaded_obj.equals(obj):
                raise ValueError(f"saved DataFrame does not match original at {f}")
        else:
            if obj != loaded_obj:
                raise ValueError(f"saved object does not match original at {f}")

    if msg:
        logger.info("saved to: %s", f)

    return f


def load(path):
    """Load a dill-serialized object from disk.

    .. warning::
        Uses ``dill.load()`` which can execute arbitrary code during
        deserialization.  Only load files from trusted sources.

    :param path: path of the file
    :return: contents of file under 'data'
    """
    f = Path(path)
    if not f.is_file():
        raise OSError(f"file: {f} not found")

    try:
        with open(f, "rb") as infile:
            file_contents = dill.load(infile)
    except RuntimeError as e:
        if "Attempting to deserialize object on a CUDA device" not in str(e):
            raise

        import torch

        logger.warning("Retrying CPU fallback load for CUDA-serialized artifact: %s", f)
        original = torch.storage._load_from_bytes

        def _cpu_load_from_bytes(blob):
            return torch.load(
                io.BytesIO(blob),
                map_location=torch.device("cpu"),
                weights_only=False,
            )

        torch.storage._load_from_bytes = _cpu_load_from_bytes
        try:
            with open(f, "rb") as infile:
                file_contents = dill.load(infile)
        finally:
            torch.storage._load_from_bytes = original

    if "data" not in file_contents:
        raise ValueError(f"contents of {f} is missing a field called `data`")
    obj = file_contents["data"]

    return obj
