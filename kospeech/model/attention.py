import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    r"""
    Applies a multi-headed scaled dot mechanism on the output features from the decoder.
    Multi-head attention proposed in "Attention Is All You Need" paper.

    Args:
        hidden_dim (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: )

    Inputs: query, value, prev_attn
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **prev_attn** (batch_size * num_heads, v_len): tensor containing previous timestep`s attention (alignment)

    Returns: context, attn
        - **context** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.

    Reference:
        - **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
        - **State-Of-The-Art Speech Recognition with Sequence-to-Sequence Models**: https://arxiv.org/abs/1712.01769
    """
    def __init__(self, hidden_dim, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim = int(hidden_dim / num_heads)
        self.query_proj = nn.Linear(hidden_dim, self.dim * num_heads)
        self.value_proj = nn.Linear(hidden_dim, self.dim * num_heads)

    def forward(self, query, value, unuse=None):
        batch_size = value.size(0)
        residual = query

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.dim)  # BxTxNxD
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.dim)  # BxTxNxD

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.dim)  # BNxTxD
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.dim)  # BNxTxD

        score = torch.bmm(query, value.transpose(1, 2)) / np.sqrt(self.dim)
        attn = F.softmax(score, dim=-1)

        context = torch.bmm(attn, value)
        context = context.view(self.num_heads, batch_size, -1, self.dim)

        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.dim)  # BxTxND
        context = torch.cat([context, residual], dim=2)

        return context, unuse


class LocationAwareAttention(nn.Module):
    """
    Applies a location-aware attention mechanism on the output features from the decoder.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    The location-aware attention mechanism is performing well in speech recognition tasks.
    We refer to implementation of ClovaCall Attention style.

    Args:
        hidden_dim (int): The number of expected features in the output
        smoothing (bool): flag indication whether to use smoothing or not.

    Inputs: query, value, last_attn
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **last_attn** (batch_size * num_heads, v_len): tensor containing previous timestep`s attention (alignment)

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the feature from encoder outputs
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.

    Reference:
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
        - **ClovaCall**: https://github.com/clovaai/ClovaCall/blob/master/las.pytorch/models/attention.py
    """
    def __init__(self, hidden_dim, smoothing=True):
        super(LocationAwareAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score_proj = nn.Linear(hidden_dim, 1, bias=True)
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.smoothing = smoothing

    def forward(self, query, value, last_attn):
        batch_size, hidden_dim, seq_len = query.size(0), query.size(2), value.size(1)

        # Initialize previous attention (alignment) to zeros
        if last_attn is None:
            last_attn = value.new_zeros(batch_size, seq_len)

        conv_attn = torch.transpose(self.conv1d(last_attn.unsqueeze(1)), 1, 2)
        score = self.score_proj(
            torch.tanh(
                self.query_proj(query.reshape(-1, hidden_dim)).view(batch_size, -1, hidden_dim)
                + self.value_proj(value.reshape(-1, hidden_dim)).view(batch_size, -1, hidden_dim)
                + conv_attn
                + self.bias
            )
        ).squeeze(dim=-1)

        if self.smoothing:
            score = torch.sigmoid(score)
            attn = torch.div(score, score.sum(dim=-1).unsqueeze(dim=-1))
        else:
            attn = F.softmax(score, dim=-1)

        context = torch.bmm(attn.unsqueeze(dim=1), value).squeeze(dim=1)  # Bx1xT X BxTxD => Bx1xD => BxD
        context = torch.cat([context, query.squeeze(1)], dim=1)

        return context, attn
