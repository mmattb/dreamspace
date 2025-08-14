import functools
import torch


def no_grad_method(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return wrapper


slerp = lambda a, b, t: (
    (
        lambda dot: (
            (
                lambda w, s: torch.where(
                    s < 1e-6,  # near-zero angle → LERP
                    (1 - t) * a + t * b,
                    (torch.sin((1 - t) * w) / s) * a + (torch.sin(t * w) / s) * b,
                )
            )(torch.acos(dot), torch.sin(torch.acos(dot)))
        )
    )(((a * b).sum(dim=-1, keepdim=True)).clamp(-1 + 1e-7, 1 - 1e-7))
)
