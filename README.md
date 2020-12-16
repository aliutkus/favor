# favor
## Simple PyTorch implementation of the FAVOR attention layer from the Performer

## Presentation

This repository implements an implementation for PyTorch of the FAVOR attention layer from the paper
> Choromanski, Krzysztof, et al. "Rethinking attention with performers." arXiv preprint arXiv:2009.14794 (2020).


The class accepts the following parameters:
```
class FAVOR(nn.Module):
    """Fast Attention Via positive Orthogonal Random features"""
    def __init__(
        self,
        key_dim, # dimension of the keys
        orthonormal=True, # whether or not the random features are drawn orthonormal
        causal=False, # whether or not to use causal ("unidirectional") attention
        m=256, # the number of random features to compute the attention
        redraw=True, # whether the features should be drawn anew each time
        kernel_fn=relu_kernel_fn, # kernel function with parameters x (data) and u (projections)
        query_kernel_fn=None, # different kernel function to use for queries; if None, defaults to kernel_fn
        scale=1., # attention scale; keys and queries will be multiplied by sqrt(scale)
        randomizer=torch.randn, # the randomizer for the features. default=gaussian
        eps=0.0, # numerical stabilizer for renormalization
        kernel_eps=0.001, # numerical stabilizer added after applying the kernel function
    )
```

The default behaviour is with the ReLU features, since they apparently perform best in the paper.

The forward function then comes as follows:

```
def forward(self, keys, values, queries):
        """
        keys: (batch, keys_dimension, *keys_locations)
        values: (batch, values_dimension, *keys_locations)
        queries: (batch, keys_dimension, *queries_locations)
        """
```

For causal attention, `keys_locations` and `queries_locations` must be equal.

## Installation

Type `pip install -e .` in the root folder of this repo.

and then
```
 from favor import FAVOR
```
