from __future__ import annotations
import torch

def apply_speed_flags(cfg_speed):
    # Core TF32 settings (faster on Ampere+ GPUs like RTX 5070 Ti)
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg_speed.get("allow_tf32", True))
    torch.backends.cudnn.allow_tf32 = bool(cfg_speed.get("cudnn_allow_tf32", True))
    # Prefer flash/mem-efficient SDPA kernels when available (PyTorch 2+)
    enable_flash = bool(cfg_speed.get("enable_flash_sdp", True))
    enable_mem_efficient = bool(cfg_speed.get("enable_mem_efficient_sdp", True))
    enable_math = bool(cfg_speed.get("enable_math_sdp", True))
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(enable_flash)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(enable_mem_efficient)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(enable_math)
    # Allow higher-throughput matmul (TF32) while keeping determinism opt-out
    torch.set_float32_matmul_precision(str(cfg_speed.get("float32_matmul_precision", "medium")))
    
    # Reduced precision reductions (optimal for RTX 50-series)
    if hasattr(torch.backends.cuda.matmul, 'allow_fp16_reduced_precision_reduction'):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = bool(
            cfg_speed.get("allow_fp16_reduced_precision_reduction", True)
        )
    
    # cuDNN optimization
    torch.backends.cudnn.benchmark = bool(cfg_speed.get("cudnn_benchmark", True))
    torch.backends.cudnn.deterministic = bool(cfg_speed.get("cudnn_deterministic", False))
