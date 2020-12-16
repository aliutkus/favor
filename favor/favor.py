from warnings import warn
import functools

import torch
from torch import nn
import torch.nn.functional as F

from favor.kernels import relu_kernel_fn, non_negative_softmax_kernel_fn


class FAVOR(nn.Module):
    """Fast Attention Via positive Orthogonal Random features"""

    def __init__(
        self,
        key_dim,
        orthonormal=True,
        causal=False,
        m=256,
        redraw=True,
        kernel_fn=relu_kernel_fn,
        query_kernel_fn=None,
        scale=1.,
        randomizer=torch.randn,
        eps=0.0,
        kernel_eps=0.001,
    ):
        super(FAVOR, self).__init__()
        self.key_dim = key_dim

        self.orthonormal = orthonormal
        self.causal = causal
        self.redraw = redraw
        self.m = m
        self.kernel_fn = kernel_fn
        self.query_kernel_fn = (query_kernel_fn if query_kernel_fn is not None
                                else kernel_fn)
        self.scale = scale
        self.randomizer = randomizer
        self.eps = eps
        self.kernel_eps = kernel_eps

        if orthonormal and m > key_dim:
            # TODO: actually we do want to allow this, see official code
            raise ValueError('m <= key_dim is required if orthonormal == True')

        self.register_buffer('proj_matrix', torch.zeros((m, key_dim)))
        self.redraw_proj_matrix()

    def redraw_proj_matrix(self):
        self.proj_matrix = self.randomizer(
            (self.key_dim, self.m),
            device=self.proj_matrix.device,
            dtype=self.proj_matrix.dtype
        )
        if self.orthonormal:
            self.proj_matrix = torch.qr(
                self.proj_matrix.double())[0].to(self.proj_matrix.dtype)

    def forward(self, keys, values, queries):
        """
        keys: (batch, keys_dimension, *keys_locations)
        values: (batch, values_dimension, *keys_locations)
        queries: (batch, keys_dimension, *queries_locations)
        """
        # flattening everything
        keys_locations = keys.shape[2:]
        queries_locations = queries.shape[2:]
        keys, values, queries = (x.view(*x.shape[:2], -1)
                                 for x in (keys, values, queries))

        if self.causal and keys_locations != queries_locations:
            raise ValueError(
                'Expected equal key and query locations with causal attention,'
                ' got: {}, {}'.format(keys_locations, queries_locations))

        # getting to (batch, n, dim)
        keys, values, queries = (x.permute(0, 2, 1)
                                 for x in (keys, values, queries))

        # multiply by sqrt(scale), so that we end up with QK^T * scale
        keys, queries = (x * self.scale ** .5 for x in (keys, queries))

        # projection matrix is (m, key_dim). randomized here if necessary
        if self.redraw:
            self.redraw_proj_matrix()

        # getting the randomized features for keys and queries
        def phi(x, kernel_fn):
            # x is (batch, n, key_dim)

            # projections are (batch, n, m)
            projections = torch.matmul(x, self.proj_matrix)

            # (batch, n, r)
            return kernel_fn(x, projections) + self.kernel_eps

        # (batch, n_context, r)
        phi_k = phi(keys, self.kernel_fn)
        # (batch, n, r)
        phi_q = phi(queries, self.query_kernel_fn)

        if self.causal:
            # outer products of keys and values: (batch, n, r, dim)
            k_v_prod = torch.matmul(
                phi_k[:, :, :, None], values[:, :, None, :])

            out = torch.matmul(         # (batch, n, dim)
                phi_q[:, :, None, :],   # (batch, n, 1, r)
                k_v_prod.cumsum(dim=1)  # (batch, n, r, dim)
            ).squeeze(2)

            # normalization factors: (batch, n, 1)
            norm = torch.matmul(
                phi_q[:, :, None, :],           # (batch, n, 1, r)
                phi_k.cumsum(dim=1)[..., None]  # (batch, n, r, 1)
            ).squeeze(2)
        else:
            out = torch.matmul(  # (batch, n, dim)
                phi_q,
                torch.matmul(  # (batch, r, dim)
                    phi_k.permute(0, 2, 1), values
                )
            )

            # normalization factors: (batch, n, 1)
            norm = torch.matmul(
                phi_q,
                phi_k.sum(dim=1)[..., None]  # (batch, r, 1)
            )

        # normalizing
        out = out / (norm + 2 * self.eps * (norm.abs() <= self.eps))

        # restoring the desired shape
        out = out.permute(0, 2, 1)
        out = out.reshape(*out.shape[:2], *queries_locations)
        return out


def make_fast_softmax_attention(key_dim, orthonormal=True, causal=False,
                                m=128, redraw=True, hyperbolic=True, eps=1e-6,
                                kernel_eps=1e-6):
    kernel_fn = functools.partial(non_negative_softmax_kernel_fn,
                                  hyperbolic=hyperbolic)
    return FAVOR(
        kernel_fn=kernel_fn, scale=key_dim ** -0.5,
        key_dim=key_dim, orthonormal=orthonormal, causal=causal, m=m,
        redraw=redraw, eps=eps, kernel_eps=kernel_eps)
