import torch
from torch import nn
from typing import Callable, Optional, Dict

def act_guardian(
    model: nn.Module,
    min_act: float = 1e-3,
    noise_scale: float = 1e-4,
    hook_handles: Optional[Dict] = None,
    verbose: bool = False
) -> Callable[[], None]:
    """
    Act-Guardian: Prevent dead neurons by injecting tiny noise if activation norm < min_act.
    
    Works best with EBC (energy_budget_clip).
    """
    if hook_handles is None:
        hook_handles = {}

    def forward_hook(module, input, output):
        module._act_guard = output.detach()

    # Register hooks
    for name, module in model.named_modules():
        if name not in hook_handles and hasattr(module, 'weight'):
            hook_handles[name] = module.register_forward_hook(forward_hook)

    def guard_fn():
        revived = 0
        total = 0
        for name, module in model.named_modules():
            if hasattr(module, '_act_guard'):
                total += 1
                act = module._act_guard
                act_norm = act.norm()
                if act_norm < min_act:
                    noise = torch.randn_like(act) * noise_scale
                    module._act_guard = act + noise
                    if hasattr(module, 'output'):
                        module.output = module.output + noise
                    revived += 1
        if verbose and revived > 0:
            print(f"[Act-Guardian] Revived {revived}/{total} layers")

    return guard_fn
