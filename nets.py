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


def get_activation(name):
    activation_map = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    if name not in activation_map:
        raise ValueError(f"Unsupported activation: {name}")
    return activation_map[name]


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
    def __init__(
        self,
        digit_embedding_dim=24,
        tower_hidden_dim=256,
        tower_hidden_layers=3,
        chunk_dim=64,
        attn_heads=2,
        ff_hidden_dim=128,
        head_hidden_dim=128,
        head_hidden_layers=1,
        activation_name="silu",
        pooling="flatten",
        use_digit_position_embedding=True,
        use_chunk_position_embedding=True,
        use_attention=True,
        use_ff=True,
    ):
        super(TowersMLPForDigit1H, self).__init__()
        if chunk_dim % attn_heads != 0:
            raise ValueError("chunk_dim must be divisible by attn_heads")
        if pooling not in {"flatten", "mean"}:
            raise ValueError("pooling must be 'flatten' or 'mean'")

        activation = get_activation(activation_name)
        tower_input_dim = 3 * digit_embedding_dim
        tower_sizes = [tower_input_dim] + [tower_hidden_dim] * tower_hidden_layers + [chunk_dim]
        head_input_dim = 3 * chunk_dim if pooling == "flatten" else chunk_dim
        head_sizes = [head_input_dim] + [head_hidden_dim] * head_hidden_layers + [1]

        self.use_digit_position_embedding = use_digit_position_embedding
        self.use_chunk_position_embedding = use_chunk_position_embedding
        self.use_attention = use_attention
        self.use_ff = use_ff
        self.pooling = pooling
        self.chunk_dim = chunk_dim
        self.digit_embedding_dim = digit_embedding_dim

        self.digit_embedding = nn.Linear(10, digit_embedding_dim, bias=False)
        self.digit_position_embedding = nn.Parameter(torch.zeros(9, digit_embedding_dim))
        self.chunk_position_embedding = nn.Parameter(torch.zeros(3, chunk_dim))
        self.tower = make_mlp(tower_sizes, activation=activation)
        self.attn_norm = nn.LayerNorm(chunk_dim)
        self.attn = nn.MultiheadAttention(chunk_dim, num_heads=attn_heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(chunk_dim)
        self.ff = nn.Sequential(
            nn.Linear(chunk_dim, ff_hidden_dim),
            activation(),
            nn.Linear(ff_hidden_dim, chunk_dim),
        )
        self.head = make_mlp(head_sizes, activation=activation)
        nn.init.normal_(self.digit_position_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.chunk_position_embedding, mean=0.0, std=0.02)

    def forward(self, x):
        batch_size = x.size(0)
        digits = x.view(batch_size, 9, 10)
        digits = self.digit_embedding(digits)
        if self.use_digit_position_embedding:
            digits = digits + self.digit_position_embedding.unsqueeze(0)
        chunks = digits.view(batch_size, 3, 3 * self.digit_embedding_dim)
        chunks = self.tower(chunks)
        if self.use_chunk_position_embedding:
            chunks = chunks + self.chunk_position_embedding.unsqueeze(0)

        if self.use_attention:
            attn_input = self.attn_norm(chunks)
            attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
            chunks = chunks + attn_output
        if self.use_ff:
            chunks = chunks + self.ff(self.ff_norm(chunks))

        if self.pooling == "mean":
            head_input = chunks.mean(dim=1)
        else:
            head_input = chunks.reshape(batch_size, -1)

        return self.head(head_input)

    


class SimpleMLPForBinary(nn.Module):
    def __init__(self):
        super(SimpleMLPForBinary, self).__init__()
        self.net = make_mlp([30, 900, 900, 900, 900, 300, 200, 20, 1], final_bias=96.0)

    def forward(self, x):
        return self.net(x)
