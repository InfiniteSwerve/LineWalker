import torch
import torch.nn as nn
import math


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Linear(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1]]
        return x


class Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(3)]
        )
        self.final_linear = nn.Linear(d_model, d_model)

    def attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_probs = nn.functional.softmax(scores, dim=-1)
        return torch.matmul(attention_probs, value)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        query, key, value = [
            l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]
        x = self.attention(query, key, value)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_k)
        )
        return self.final_linear(x)


class MLP(nn.Module):
    def __init__(self, d_model, d_hidden, act_fn, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.w_2 = nn.Linear(d_hidden, d_model)
        # self.dropout = nn.Dropout(dropout)
        self.activation = act_fn

    def forward(self, x):
        # return self.w_2(self.dropout(self.activation(self.w_1(x))))
        return self.w_2(self.activation(self.w_1(x)))


class Block(nn.Module):
    def __init__(self, d_model, num_heads, d_hidden, dropout, act_fn):
        super().__init__()
        self.attn = Attention(d_model, num_heads)
        self.mlp = MLP(d_model, d_hidden, act_fn, dropout)
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.mlp_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out = self.attn(x, x, x)
        x = x + attn_out
        x = self.attn_layer_norm(x)

        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.mlp_layer_norm(x)

        return x


class GPT(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_hidden,
        input_dim,
        max_seq_len=512,
        dropout=0.1,
        act_fn=nn.ReLU(),
    ):
        super().__init__()
        self.embedding = Embedding(input_dim, d_model)
        self.positional_encoding = PositionalEmbedding(d_model, max_seq_len)
        self.blocks = nn.ModuleList(
            [
                Block(d_model, num_heads, d_hidden, dropout, act_fn)
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.unembedding = nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.blocks:
            x = layer(x)  # For GPT, the input and memory are the same

        x = self.layer_norm(x)
        logits = self.unembedding(x)

        return logits
