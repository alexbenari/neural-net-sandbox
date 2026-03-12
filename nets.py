import torch
import torch.nn as nn


class _MLP(nn.Module):
    def __init__(self, sizes, activation, final_bias, bias):
        super().__init__()
        if len(sizes) < 2:
            raise ValueError("sizes must include at least input and output size.")
        self.layers = nn.ModuleList(
            nn.Linear(in_size, out_size, bias=bias)
            for in_size, out_size in zip(sizes[:-1], sizes[1:])
        )
        if final_bias is not None and bias:
            nn.init.constant_(self.layers[-1].bias, float(final_bias))
        self.activation = None
        self.activations = None
        if activation is None:
            return
        if isinstance(activation, nn.Module):
            self.activation = activation
        elif isinstance(activation, type) and issubclass(activation, nn.Module):
            self.activations = nn.ModuleList(
                activation() for _ in range(len(self.layers) - 1)
            )
        else:
            raise TypeError(
                "activation must be an nn.Module instance, an nn.Module class, or None."
            )

    def forward(self, x):
        for idx, layer in enumerate(self.layers[:-1]):
            if self.activations is not None:
                x = self.activations[idx](layer(x))
            elif self.activation is not None:
                x = self.activation(layer(x))
            else:
                x = layer(x)
        return self.layers[-1](x)


def make_mlp(sizes, activation=nn.ReLU, final_bias=0.0, bias=True):
    return _MLP(sizes, activation, final_bias, bias)


class SimpleMLPForInt(nn.Module):
    def __init__(self):
        super(SimpleMLPForInt, self).__init__()
        self.net = make_mlp([1, 900, 900, 900, 900, 300, 200, 20, 1], final_bias=96.0)

    def forward(self, x):
        return self.net(x)


class SimpleMLPForDigits(nn.Module):
    def __init__(self):
        super(SimpleMLPForDigits, self).__init__()
        self.net = make_mlp([9, 900, 900, 900, 900, 300, 200, 20, 1], final_bias=96.0)

    def forward(self, x):
        return self.net(x)


class SimpleMLPForDigit1H(nn.Module):
    def __init__(self):
        super(SimpleMLPForDigit1H, self).__init__()
        #self.net = make_mlp([90, 900, 900, 900, 900, 300, 200, 20, 1], final_bias=96.0)
        #self.net = make_mlp([90, 270, 256, 64, 1], final_bias=96.0)
        #self.net = make_mlp([90, 500, 500, 256, 1], final_bias=96.0)
        #self.net = make_mlp([90, 500, 500, 256, 256, 64, 64, 1], final_bias=96.0)
        #self.net = make_mlp([90, 1000, 500, 256, 256, 64, 64, 1], final_bias=96.0)
        self.net = make_mlp([90, 10000, 500, 256, 256, 64, 64, 1], final_bias=96.0)
        #self.net = make_mlp([90, 50000, 500, 256, 256, 64, 64, 1], final_bias=96.0)

    def forward(self, x):
        return self.net(x)


class TowersMLPForDigit1H(nn.Module):
    def __init__(self):
        super(TowersMLPForDigit1H, self).__init__()
        self.digit_embedding = nn.Linear(10, 24, bias=False)
        self.digit_position_embedding = nn.Parameter(torch.zeros(9, 24))
        self.chunk_position_embedding = nn.Parameter(torch.zeros(3, 64))
        self.tower = make_mlp([72, 256, 256, 64], activation=nn.SiLU)
        self.attn_norm = nn.LayerNorm(64)
        self.attn = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
        self.ff_norm = nn.LayerNorm(64)
        self.ff = nn.Sequential(
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
        )
        self.head = make_mlp([192, 160, 64, 1], activation=nn.SiLU)
        nn.init.normal_(self.digit_position_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.chunk_position_embedding, mean=0.0, std=0.02)

    def forward(self, x):
        batch_size = x.size(0)
        digits = x.view(batch_size, 9, 10)
        digits = self.digit_embedding(digits)
        digits = digits + self.digit_position_embedding.unsqueeze(0)
        chunks = digits.view(batch_size, 3, 72)
        chunks = self.tower(chunks)
        chunks = chunks + self.chunk_position_embedding.unsqueeze(0)

        attn_input = self.attn_norm(chunks)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        chunks = chunks + attn_output
        chunks = chunks + self.ff(self.ff_norm(chunks))

        return self.head(chunks.reshape(batch_size, -1))

    


class SimpleMLPForBinary(nn.Module):
    def __init__(self):
        super(SimpleMLPForBinary, self).__init__()
        self.net = make_mlp([30, 900, 900, 900, 900, 300, 200, 20, 1], final_bias=96.0)

    def forward(self, x):
        return self.net(x)
