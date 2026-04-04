import gc
import logging
import os

import torch

logger = logging.getLogger(__name__)

def cleanup_files(*paths):
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError as e:
                logger.warning("Failed to delete %s: %s", path, e)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()