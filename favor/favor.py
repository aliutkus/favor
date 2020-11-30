import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from warnings import warn
import itertools
import ipdb
import math
import numpy as np


class FAVOR(nn.Module):
    """Fast Attention Via positive Orthogonal Random features"""
    def __init__(
        self,
        key_dim,
        orthonormal=True,
        m=128,
        redraw=True,
        h=lambda x: 1.,
        f=[F.relu,],
        randomizer=torch.randn,
    ):
        super(FAVOR, self).__init__()
        self.key_dim = key_dim

        self.orthonormal=orthonormal
        self.redraw=redraw
        self.m = m
        self.h = h
        self.f = f
        self.randomizer=randomizer

        if orthonormal and m > key_dim:
            raise ValueError('m <= key_dim is required if orthonormal == True')

        self._features = None
        self.register_buffer('phi_scale', torch.tensor(1./ math.sqrt(m)))


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
        values: (batch, values_dimension, *values_locations)
        queries: (batch, keys_dimension, *queries_locations)
        """
        # flattening everything
        keys_locations = keys.shape[2:]
        queries_locations = queries.shape[2:]
        keys = keys.view(*keys.shape[:2], -1)
        values = values.view(*values.shape[:2], -1)
        queries = queries.view(*queries.shape[:2], -1)

        # getting to (batch, n, dim)
        keys = keys.permute(0, 2, 1)
        values = values.permute(0, 2, 1)
        queries = queries.permute(0, 2, 1)

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
                dim = -1
            ) * self.h(x) * self.phi_scale

        # (batch, n_context, r)
        phi_k = phi(keys)
        # (batch, n, r)
        phi_q = phi(queries)

        out = torch.matmul( # (batch, n, dim)
            phi_q,
            torch.matmul( # (batch, r, dim)
                phi_k.permute(0, 2, 1), values
            )
        )

        # rescaling: (batch, n, 1)
        scale = torch.matmul(
            phi_q,
            torch.sum(phi_k, dim=1)[..., None] # (batch, r, 1)
        ) 

        out = out / scale

        # restoring the desired shape
        out = out.permute(0, 2, 1)
        out = out.reshape(*out.shape[:2], *queries_locations)
        return out
