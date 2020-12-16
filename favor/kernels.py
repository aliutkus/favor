import torch
import torch.nn.functional as F


def relu_kernel_fn(x, u):
    """Kernel function with f_1 = ReLU and h(x) = sqrt(m)."""
    del x
    return F.relu(u)


def non_negative_softmax_kernel_fn(x, u, hyperbolic=True):
    """Non-negative (hyperbolic) softmax kernel function.

    For hyperbolic=True:
        f_1(u) = exp(u),
        f_2(u) = exp(-u),
        h(x) = 1 / sqrt(2) * exp(-|x|^2 / 2).

    For hyperbolic=False:
        f_1(u) = exp(u),
        h(x) = exp(-|x|^2 / 2).
    """
    if hyperbolic:
        # u and -u, shape (batch, n, 2m)
        u = torch.cat([u, -u], dim=-1)
    # |x|^2, shape (batch, n, 1)
    x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)

    # TODO: the official implementation subtracts u.max() here for keys and
    # u.max(dim=-1, keepdims=True) for queries, apparently for "numerical
    # stability"
    out = torch.exp(u - x_norm_sq / 2)

    # normalize by sqrt(m), or sqrt(2m) if using hyperbolic kernel
    return out / u.shape[-1] ** .5
