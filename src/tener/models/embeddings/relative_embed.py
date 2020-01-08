from tener.misc.pretty_print import print_error, print_info

__author__ = "Mageswaran Dhandapani"
__copyright__ = "Copyright 2020, The Dhira Project"
__credits__ = []
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Mageswaran Dhandapani"
__email__ = "mageswaran1989@gmail.com"
__status__ = "Developement"

"""
Relative Positional Embeddings
"""

import tensorflow as tf

import torch
from torch import nn
import torch.nn.functional as F
import math

class RelativeSinusoidalPositionalEmbedding(tf.keras.layers.Layer):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        """
        :param embedding_dim:
        :param padding_idx:
        :param init_size:
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size % 2 == 0
        self.embd_weights = self.get_embedding(
            init_size + 1,
            embedding_dim,
            padding_idx,
        )
        # self.register_buffer('weights', weights)
        # self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def get_embedding(self, num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        if padding_idx is not None:
            num_embeddings = num_embeddings - 1

        half_dim = embedding_dim // 2

        emb = tf.math.log(10000.) / (half_dim - 1)
        emb = tf.math.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = tf.expand_dims(tf.range(-num_embeddings//2, num_embeddings//2, dtype=tf.float32), 1) * tf.expand_dims(emb, 0)
        emb = tf.reshape(tf.concat([tf.math.sin(emb), tf.math.cos(emb)], axis=1), (num_embeddings, -1))

        # TODO Tensorflow doesn't allow inplace assignment
        if padding_idx is not None:
            emb = tf.concat([tf.zeros((1, embedding_dim)), emb], axis=0)

        if embedding_dim % 2 == 1:
            # zero pad
            emb = tf.concat([emb, tf.zeros(num_embeddings, 1)], axis=1)

        self.origin_shift = num_embeddings//2 + 1
        return emb

    def call(self, input: tf.Tensor):
        """

        :param input: Tensor of size [batch_size x seq_length]
        :return:
        """
        if len(input.shape) == 4:
            batch_size, _, _, seq_len = input.shape
        else:
            batch_size, seq_len = input.shape
        max_pos = self.padding_idx + seq_len

        if max_pos > self.origin_shift:
            # recompute/expand embeddings if needed
            self.embd_weights = self.get_embedding(
                max_pos*2,
                self.embedding_dim,
                self.padding_idx,
            )
            self.origin_shift = self.embd_weights.size(0)//2

        positions = tf.range(-seq_len, seq_len) + self.origin_shift  # 2 * seq_len
        embed = tf.gather(self.embd_weights, positions)
        return embed


# ======================================================================================================================

class RelativeEmbedding(nn.Module):
    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen].
        """
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + seq_len
        if max_pos > self.origin_shift:
            # recompute/expand embeddings if needed
            weights = self.get_embedding(
                max_pos*2,
                self.embedding_dim,
                self.padding_idx,
                )
            weights = weights.to(self._float_tensor)
            del self.weights
            self.origin_shift = weights.size(0)//2
            self.register_buffer('weights', weights)

        positions = torch.arange(-seq_len, seq_len).to(input.device).long() + self.origin_shift  # 2*seq_len
        embed = self.weights.index_select(0, positions.long()).detach()
        return embed


class RelativeSinusoidalPositionalEmbeddingTorch(RelativeEmbedding):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        """

        :param embedding_dim: 每个位置的dimension
        :param padding_idx:
        :param init_size:
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size%2==0
        weights = self.get_embedding(
            init_size+1,
            embedding_dim,
            padding_idx,
            )
        self.register_buffer('weights', weights)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def get_embedding(self, num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(-num_embeddings//2, num_embeddings//2, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        self.origin_shift = num_embeddings//2 + 1
        return emb