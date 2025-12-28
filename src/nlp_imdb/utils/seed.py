# src/nlp_imdb/utils/seed.py

import random
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def set_global_seed(seed: int) -> None:
    """
    Set seed for reproducibility across common libraries.
    """
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
