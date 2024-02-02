import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


### Encoder ###
class Encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding_dim = 512
        self.num_heads = 8
        self.head_dim = int(
            self.embedding_dim / self.num_heads
        )  # d_k = d_v = d_model/h=64

        self.mha = nn.MultiheadAttention(
            self.head_dim, self.num_heads
        )  # TODO: implement a custom version of this to fully understand
        self.ff = nn.Linear(
            self.embedding_dim, self.embedding_dim
        )  # All sub-layers in the model as well as embedding layers output dimension d_model = 512
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(self, x: torch.Tensor):
        """Passes the input embedding through the encoder

        Broken into each step for clarity, not for optimization

        Args:
            x (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        x1 = self.mha(x)
        # its not immediately clear what to put for the key and value vectors here
        x2 = self.layer_norm(x + x1)
        x3 = self.ff(x2)
        x4 = self.layer_norm(x2 + x3)
        return x4


class Transformer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass


def test():
    test_str = "A cat sat on my lap."
    test_emb = torch.rand(512)
    encoder = Encoder()
    res = encoder.forward(test_emb)
    print(res)


if __name__ == "__main__":
    test()
