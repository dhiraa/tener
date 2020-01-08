import math

import numpy as np
import tensorflow as tf

import torch
from torch import nn
import torch.nn.functional as F
import math

from tener.misc.pretty_print import print_info


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def make_positions(tensor, padding_idx=0):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tf.math.not_equal(tensor, padding_idx)
    mask = tf.cast(mask, tf.int32)
    return (
                   tf.math.cumsum(mask, axis=1) * mask
           ) + padding_idx


class SinusoidalPositionalEmbeddingNaive(tf.keras.layers.Layer):
    """
    PE_(pos, 2i) = sin(pos/10000^(2i/d_model)) # even position
    PE_(pos, 2i+1) = cos(pos/10000^(2i/d_model)) # odd position
    """
    def __init__(self, maximum_position_encoding, d_model):
        # super(SinusoidalPositionalEmbedding, self).__init__()
        super().__init__()
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

    def call(self, x):
        """

        :param x: Tensor of soze [batch_size, max_seq_length]
        :return:
        """
        max_seq_len = tf.shape(x)[1]
        pos = self.pos_encoding[:, :max_seq_len, :]
        return pos


class SinusoidalPositionalEmbedding(tf.keras.layers.Layer):
    """
    This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embd_weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.embd_weights = tf.convert_to_tensor(self.embd_weights)

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim, dtype=np.float32) * -emb)
        emb = np.expand_dims(np.arange(num_embeddings, dtype=np.float32), 1) * np.expand_dims(emb, 0)
        emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=1)
        emb = np.reshape(emb, (num_embeddings, -1))

        if embedding_dim % 2 == 1:
            # zero pad
            emb = np.concatenate([emb, np.zeros(num_embeddings, 1)], axis=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def call(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.shape
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.embd_weights.shape[0]:
            # recompute/expand embeddings if needed
            self.embd_weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.embd_weights = tf.convert_to_tensor(self.embd_weights)
        positions = make_positions(input, self.padding_idx)
        embed = tf.gather(self.embd_weights, positions)
        return embed

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class LearnedPositionalEmbedding(tf.keras.layers.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int = 0):
        #TODO mask zero is under assumption that 0 is always used as padding index
        super().__init__(input_dim=num_embeddings, output_dim=embedding_dim, mask_zero=True)
        self.padding_idx = padding_idx

    def call(self, x):
        # positions: batch_size x max_len, 把words的index输入就好了
        positions = make_positions(x, self.padding_idx)
        return super().call(positions)

# ======================================================================================================================


def make_positions_torch(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
                   torch.cumsum(mask, dim=1).type_as(mask) * mask
           ).long() + padding_idx

class SinusoidalPositionalEmbeddingTorch(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbeddingTorch.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbeddingTorch.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        positions = make_positions_torch(input, self.padding_idx)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number
