import tensorflow as tf

from tener.misc.pretty_print import print_error, print_info, print_warn
from tener.models.embeddings.relative_embed import RelativeSinusoidalPositionalEmbedding, \
    RelativeSinusoidalPositionalEmbeddingTorch

import torch
from torch import nn
import torch.nn.functional as F
import math


class RelativeMultiHeadAttn(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head, dropout, batch_size=None, r_w_bias=None, r_r_bias=None, scale=False):
        """
        :param int d_model:
        :param int n_head:
        :param dropout: 对attention map的dropout
        :param r_w_bias: n_head x head_dim or None, 如果为dim
        :param r_r_bias: n_head x head_dim or None,
        :param scale:
        :param rel_pos_embed:
        """
        super().__init__()

        self._batch_size = batch_size
        self.qv_linear = tf.keras.layers.Dense(d_model * 2, use_bias=False) #nn.Linear(d_model, d_model * 2, bias=False)
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout_layer = tf.keras.layers.Dropout(dropout)

        self.pos_embed = RelativeSinusoidalPositionalEmbedding(d_model // n_head, 0, 1200)

        if scale:
            self.scale = tf.math.sqrt(d_model // n_head)
        else:
            self.scale = 1

        if r_r_bias is None or r_w_bias is None:  # Biases are not shared
            initializer = tf.initializers.GlorotUniform()
            self.r_r_bias = tf.Variable(initializer(shape=(n_head, d_model // n_head)))
            # #nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, d_model // n_head)))
            self.r_w_bias = tf.Variable(initializer(shape=(n_head, d_model // n_head)))
            # nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, d_model // n_head)))
        else:
            self.r_r_bias = r_r_bias  # r_r_bias就是v
            self.r_w_bias = r_w_bias  # r_w_bias就是u

    def _shift(self, BD):
        """
        Input Similar to:
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        Translate to:
         0  1  2
        -1  0  1
        -2 -1  0
        :param BD: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        bsz, n_head, max_len, _ = BD.shape
        zero_pad = tf.zeros([bsz, n_head, max_len, 1])
        BD = tf.concat([BD, zero_pad], axis=-1)
        BD = tf.reshape(BD, (bsz, n_head, -1, max_len))  # bsz x n_head x (2max_len+1) x max_len
        BD = BD[:, :, :-1]
        BD = tf.reshape(BD, (bsz, n_head, max_len, -1))  # bsz x n_head x 2max_len x max_len
        BD = BD[:, :, :, max_len:]

        return BD

    def call(self, x, mask):
        """
        :param x: batch_size x max_len x d_model
        :param mask: batch_size x max_len
        :return:
        """
        batch_size, max_len, d_model = x.shape
        pos_embed = self.pos_embed(mask)  # l x head_dim

        qv = self.qv_linear(x)  # batch_size x max_len x d_model2

        q, v = tf.split(qv, 2, axis=-1)

        q = tf.reshape(q, (-1, max_len, self.n_head, d_model // self.n_head))
        q = tf.transpose(q, perm=[0, 2, 1, 3])

        k = tf.reshape(x, (-1, max_len, self.n_head, d_model // self.n_head))
        k = tf.transpose(k, perm=[0, 2, 1, 3])

        v = tf.reshape(v, (-1, max_len, self.n_head, d_model // self.n_head))  # [batch_size, max_seq_len, num_heads, 16]b x n x l x d
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        rw_head_q = q + self.r_r_bias[:, None]
        AC = tf.einsum('bnqd,bnkd->bnqk', rw_head_q, k)  # b x n x l x d, n是head

        D_ = tf.einsum('nd,ld->nl', self.r_w_bias, pos_embed)[None, :, None]  # head x 2max_len, 每个head对位置的bias
        B_ = tf.einsum('bnqd,ld->bnql', q, pos_embed)  # bsz x head  x max_len x 2max_len，每个query对每个shift的偏移
        BD = B_ + D_  # bsz x head x max_len x 2max_len, 要转换为bsz x head x max_len x max_len
        BD = self._shift(BD)
        attn = AC + BD

        attn = attn / self.scale
        attn = attn * mask
        # attn = attn.masked_fill(mask[:, None, None, :].eq(0), float('-inf')) TODO
        attn = tf.where(attn == 0, float('-inf'), attn)

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.dropout_layer(attn)

        v = tf.matmul(attn, v)

        # v = tf.transpose(v, [1, 2])
        v = tf.transpose(v, [0, 2, 1, 3])
        v = tf.reshape(v, [-1, max_len, d_model])  # b x n x l x d

        return v

# =======================================================================================================================


class RelativeMultiHeadAttnTorch(nn.Module):
    def __init__(self, d_model, n_head, dropout, r_w_bias=None, r_r_bias=None, scale=False):
        """

        :param int d_model:
        :param int n_head:
        :param dropout: 对attention map的dropout
        :param r_w_bias: n_head x head_dim or None, 如果为dim
        :param r_r_bias: n_head x head_dim or None,
        :param scale:
        :param rel_pos_embed:
        """
        super().__init__()
        self.qv_linear = nn.Linear(d_model, d_model * 2, bias=False)
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout_layer = nn.Dropout(dropout)

        self.pos_embed = RelativeSinusoidalPositionalEmbeddingTorch(d_model//n_head, 0, 1200)

        if scale:
            self.scale = math.sqrt(d_model // n_head)
        else:
            self.scale = 1

        if r_r_bias is None or r_w_bias is None:  # Biases are not shared
            self.r_r_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, d_model // n_head)))
            self.r_w_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, d_model // n_head)))
        else:
            self.r_r_bias = r_r_bias  # r_r_bias就是v
            self.r_w_bias = r_w_bias  # r_w_bias就是u

    def forward(self, x, mask):
        """

        :param x: batch_size x max_len x d_model
        :param mask: batch_size x max_len
        :return:
        """

        batch_size, max_len, d_model = x.size()
        pos_embed = self.pos_embed(mask)  # l x head_dim

        qv = self.qv_linear(x)  # batch_size x max_len x d_model*2

        q, v = torch.chunk(qv, chunks=2, dim=-1)

        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = x.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)  # b x n x l x d

        rw_head_q = q + self.r_r_bias[:, None]
        AC = torch.einsum('bnqd,bnkd->bnqk', [rw_head_q, k])  # b x n x l x d, n是head

        D_ = torch.einsum('nd,ld->nl', self.r_w_bias, pos_embed)[None, :, None]  # head x 2max_len, 每个head对位置的bias
        B_ = torch.einsum('bnqd,ld->bnql', q, pos_embed)  # bsz x head  x max_len x 2max_len，每个query对每个shift的偏移
        BD = B_ + D_  # bsz x head x max_len x 2max_len, 要转换为bsz x head x max_len x max_len
        BD = self._shift(BD)
        attn = AC + BD

        attn = attn / self.scale

        attn = attn.masked_fill(mask[:, None, None, :].eq(0), float('-inf'))


        attn = F.softmax(attn, dim=-1)

        attn = self.dropout_layer(attn)

        v = torch.matmul(attn, v)
        v = v.transpose(1, 2).reshape(batch_size, max_len, d_model)  # b x n x l x d

        return v

    def _shift(self, BD):
        """
        类似
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2

        转换为
        0   1  2
        -1  0  1
        -2 -1  0

        :param BD: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        bsz, n_head, max_len, _ = BD.size()
        zero_pad = BD.new_zeros(bsz, n_head, max_len, 1)
        BD = torch.cat([BD, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)  # bsz x n_head x (2max_len+1) x max_len
        BD = BD[:, :, :-1].view(bsz, n_head, max_len, -1)  # bsz x n_head x 2max_len x max_len
        BD = BD[:, :, :, max_len:]
        return BD

"""
ERROR:tensorflow:Tensor("tener_keras_model/tener_encoder/transformer_attn_layer/relative_multi_head_attn/concat:0", shape=(16, 8, 22, 45), dtype=float32)
ERROR:tensorflow:Tensor("tener_keras_model/tener_encoder/transformer_attn_layer/relative_multi_head_attn/Reshape_3:0", shape=(16, 8, 45, 22), dtype=float32)
ERROR:tensorflow:Tensor("tener_keras_model/tener_encoder/transformer_attn_layer/relative_multi_head_attn/strided_slice_2:0", shape=(16, 8, 44, 22), dtype=float32)
ERROR:tensorflow:Tensor("tener_keras_model/tener_encoder/transformer_attn_layer/relative_multi_head_attn/Reshape_4:0", shape=(16, 8, 22, 44), dtype=float32)
ERROR:tensorflow:Tensor("tener_keras_model/tener_encoder/transformer_attn_layer/relative_multi_head_attn/strided_slice_3:0", shape=(16, 8, 22, 22), dtype=float32)

"""