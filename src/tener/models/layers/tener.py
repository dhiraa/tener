# import sys
# sys.path.append("/opt/vlab/tener/src/")

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tener.utils import CustomSchedule
from tener.models.model_utils import create_masks, get_sequence_length, create_look_ahead_mask, create_padding_mask
from tener.models.attention.multihead_naive_attn import MultiHeadAttention
from tener.models.attention.multihead_relative_attn import RelativeMultiHeadAttn
from tener.models.embeddings.sinusoidal_embd import positional_encoding
from tener.misc.pretty_print import print_error
from tener.models.embeddings import character_embd# import TransformerCharEncoding


"""
Transformer modules
- MultiHead Attention
- Encoder Layer
- Decoder Layer
- Encoder
- Decoder
- Transformer
"""


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class TransformerAttnLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, attn_type="relative"):
        super(TransformerAttnLayer, self).__init__()
        self._attn_type = attn_type
        if attn_type == "naive":
            self.self_attn = MultiHeadAttention(d_model, num_heads)
        elif attn_type == "relative":
            self.self_attn = RelativeMultiHeadAttn(d_model=d_model,
                                                   n_head=num_heads,
                                                   dropout=0.5,
                                                   r_w_bias=None,
                                                   r_r_bias=None,
                                                   scale=False)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        """

        :param x:
        :param training:
        :param mask:
        :return:
        """
        if self._attn_type == "naive":
            attn_output, _ = self.self_attn(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        else:
            attn_output = self.self_attn(x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class TenerEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 input_vocab_size,
                 target_vocab_size,
                 rate=0.1,
                 is_pos_emb=True):
        super(TenerEncoder, self).__init__()

        self.d_model = d_model
        self.is_pos_emb = is_pos_emb
        self.num_layers = num_layers

        self.enc_layers = [TransformerAttnLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, mask, training):
        """

        :param x:
        :param mask:
        :param training:
        # :return:
        """
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x  # (batch_size, input_seq_len, d_model)


class TenerKerasModel(tf.keras.Model):
    def __init__(self,
                 num_layers,
                 word_d_model,
                 char_d_model,
                 num_heads,
                 dff,
                 input_vocab_size,
                 target_vocab_size,
                 is_char_embd=True,
                 rate=0.1):
        super(TenerKerasModel, self).__init__()


        self._word_d_model = word_d_model

        if is_char_embd:
            self.d_model = word_d_model + char_d_model
        else:
            self.d_model = word_d_model

        if is_char_embd:
            self._char_embd = character_embd.TransformerCharEncoding(char_emd_dim=char_d_model,
                                                                     len_char_vocab=100,
                                                                     d_model=char_d_model,
                                                                     target_vocab_size=target_vocab_size)  # TODO length
        else:
            self._char_embd = None

        self._word_embedding = tf.keras.layers.Embedding(input_vocab_size, word_d_model)
        self.sin_pos_encoding = positional_encoding(input_vocab_size, self._word_d_model)

        self.enc_layers = TenerEncoder(num_layers=num_layers,
                                       d_model=self.d_model,
                                       num_heads=num_heads,
                                       dff=dff,
                                       input_vocab_size=input_vocab_size,
                                       target_vocab_size=target_vocab_size,
                                       rate=0.1,
                                       is_pos_emb=True)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, x):
        """

        :param x: Tensor [batch_size, max_seq_length, d_model/embd_size]
        :param y: Tensor[batch_size,max_seq_length]
        :param mask:
        :param training:
        :return:
        """

        # print_error(x)
        word_ids = x["word_ids"]  # [bacth_size, max_seq_length]
        char_ids = x["char_ids"]

        word_embedding = self._word_embedding(word_ids)  # [bacth_size, max_seq_length, d_model/word_embd_size]
        word_embedding *= tf.math.sqrt(tf.cast(self._word_d_model, tf.float32))

        seq_len = tf.shape(word_embedding)[1]

        pos = self.sin_pos_encoding[:, :seq_len, :]  # (1, 40, 128) broadcasting same position for entire batch
        word_embedding += pos

        if self._char_embd:
            char_embedding = self._char_embd(char_ids)
            embedding = tf.concat([word_embedding, char_embedding], axis=-1)
        else:
            embedding = word_embedding

        word_mask = create_padding_mask(word_ids)

        encoded = self.enc_layers(embedding, word_mask, True)
        final_output = self.final_layer(encoded)
        return final_output