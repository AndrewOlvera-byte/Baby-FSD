from __future__ import annotations
import torch

def apply_speed_flags(cfg_speed):
    # Core TF32 settings (faster on Ampere+ GPUs like RTX 5070 Ti)
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg_speed.get("allow_tf32", True))
    torch.backends.cudnn.allow_tf32 = bool(cfg_speed.get("cudnn_allow_tf32", True))
    
    # Reduced precision reductions (optimal for RTX 50-series)
    if hasattr(torch.backends.cuda.matmul, 'allow_fp16_reduced_precision_reduction'):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = bool(
            cfg_speed.get("allow_fp16_reduced_precision_reduction", True)
        )
    
    # cuDNN optimization
    torch.backends.cudnn.benchmark = bool(cfg_speed.get("cudnn_benchmark", True))
    torch.backends.cudnn.deterministic = bool(cfg_speed.get("cudnn_deterministic", False))
