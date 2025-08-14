import functools
import torch


def no_grad_method(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return wrapper


def slerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    a, b: (..., D) unit vectors (will be normalized here just in case)
    t:    () scalar or (K,) or broadcastable; we auto-expand to (..., 1) along last dim
    returns: (broadcast of t over a/b, ..., D)
    """
    # normalize (safe)
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # make t shape (K, 1, 1, ..., 1) so last dim aligns with D
    if t.ndim == 0:
        t_exp = t
    else:
        expand_dims = (1,) * (a.ndim - 1)  # keep last dim for features
        t_exp = t.view(t.shape + expand_dims)

    dot = (a * b).sum(dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    w = torch.acos(dot)  # (..., 1)
    s = torch.sin(w)  # (..., 1)

    # near-zero angle → lerp fallback
    lerp = (1 - t_exp) * a + t_exp * b
    slerp = (torch.sin((1 - t_exp) * w) / s) * a + (torch.sin(t_exp * w) / s) * b
    return torch.where(s < 1e-6, lerp, slerp)
