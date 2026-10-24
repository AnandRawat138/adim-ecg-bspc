import random
import os
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility across python, numpy, and pytorch.

    Args:
        seed: The random seed to use.
        deterministic: Whether to set cudnn deterministic flag and disable benchmarking.
    """
    logger.info(f"Setting random seed: {seed} (deterministic={deterministic})")
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # If supported by the environment, try to use deterministic ops
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            logger.info("Deterministic algorithms enabled (warn_only=True)")
        except AttributeError:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
