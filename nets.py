import torch
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
        #self.tower = make_mlp([30, 768, 1024, 512])
        #self.tower = make_mlp([30, 30, 30, 30, 30, 30 , 30 , 20])
        #self.tower = make_mlp([30, 30, 100, 300, 100, 30 , 30 , 30])
        #self.tower = make_mlp([30, 100, 200, 300, 200, 100 , 100 , 30])
        self.tower = make_mlp([30, 300, 500, 500, 300, 300 , 100 , 30])
        #self.head = make_mlp([1536, 512, 128, 1])
        self.head = make_mlp([90, 1024, 512, 256, 128, 1])
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Split x into 3 vectors of 30.
        x1 = x[:, 0:30]
        x2 = x[:, 30:60]
        x3 = x[:, 60:90]
        # Pass each vector through the shared tower and concat the result.
        t1 = self.tower(x1)
        t2 = self.tower(x2)
        t3 = self.tower(x3)
        t = torch.cat((t1, t2, t3), dim=1)
        #t = self.dropout(t)
        return self.head(t)

    


class SimpleMLPForBinary(nn.Module):
    def __init__(self):
        super(SimpleMLPForBinary, self).__init__()
        self.net = make_mlp([30, 900, 900, 900, 900, 300, 200, 20, 1], final_bias=96.0)

    def forward(self, x):
        return self.net(x)
