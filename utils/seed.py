"""Deterministic seed helpers."""
import hashlib
import random

import numpy as np
import torch


def per_case_epoch_seed(case_id, epoch: int) -> int:
    """Deterministic 32-bit seed from (case_id, epoch).

    ``case_id`` can be either an int (legacy index) or a str (case_name,
    current convention); the function stringifies and hashes either way.
    """
    h = hashlib.blake2b(f'{case_id}:{epoch}'.encode(), digest_size=4)
    return int(h.hexdigest(), 16)


def make_rng(seed_int: int) -> np.random.Generator:
    return np.random.default_rng(seed_int)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
