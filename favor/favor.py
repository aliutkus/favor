import torch
from torch import nn
import torch.nn.functional as F
from warnings import warn
import itertools
import math
import numpy as np


class FAVOR(nn.Module):
    """Fast Attention Via positive Orthogonal Random features"""

    def __init__(
        self,
        key_dim,
        orthonormal=True,
        causal=False,
        m=128,
        redraw=True,
        h=None,
        f=[F.relu],
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
        sqrt_m = math.sqrt(m)
        self.h = h if h is not None else lambda x: sqrt_m
        self.f = f
        self.randomizer = randomizer
        self.eps = eps
        self.kernel_eps = kernel_eps

        if orthonormal and m > key_dim:
            raise ValueError('m <= key_dim is required if orthonormal == True')

        self._features = None
        self.register_buffer('phi_scale', torch.tensor(1. / sqrt_m))

    def features(self):
        if self._features is None or self.redraw:
            self._features = self.randomizer(
                (self.key_dim, self.m),
                device=self.phi_scale.device,
                dtype=self.phi_scale.dtype
            )
            if self.orthonormal:
                self._features = torch.qr(
                    self._features.double())[0].to(self.phi_scale.dtype)
            self._features.t_()
        return self._features

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

        # features are (m, key_dim). randomized here if necessary
        features = self.features()

        # getting the randomized features for keys and queries
        def phi(x):
            # x is (batch, n, key_dim)

            # projections are (batch, n, m)
            projections = torch.matmul(x, features.T)

            # (batch, n, r)
            return torch.cat(
                [f(projections) for f in self.f],
                dim=-1
            ) * self.h(x) * self.phi_scale + self.kernel_eps

        # (batch, n_context, r)
        phi_k = phi(keys)
        # (batch, n, r)
        phi_q = phi(queries)

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
