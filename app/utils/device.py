import logging

import torch

logger = logging.getLogger(__name__)

def setup_device() -> tuple[str, str, bool]:
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(0), gpu_memory)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if gpu_memory >= 8:
            model_size = "large"
        elif gpu_memory >= 6:
            model_size = "medium"
        elif gpu_memory >= 4:
            model_size = "small"
        else:
            model_size = "base"

        return "cuda", model_size, True

    if torch.backends.mps.is_available():
        logger.info("Device: Apple MPS")
        return "mps", "medium", False

    logger.info("Device: CPU")
    return "cpu", "base", False