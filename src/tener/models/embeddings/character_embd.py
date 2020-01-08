import tensorflow as tf
from tener.misc.pretty_print import print_info
from tener.models.model_utils import create_padding_mask
from tener.models.layers import tener


class TransformerCharEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, char_emd_dim, len_char_vocab, target_vocab_size):
        super(TransformerCharEncoding, self).__init__()
        self._char_emb_dim = char_emd_dim
        self._len_char_vocab = len_char_vocab

        self._encoder = tener.TenerEncoder(num_layers=4,
                                           d_model=d_model,
                                           num_heads=4,
                                           dff=512,
                                           input_vocab_size=len_char_vocab,
                                           rate=0.1,
                                           # is_pos_emb=False,
                                           target_vocab_size=target_vocab_size)

        self._char_embed_layer = tf.keras.layers.Embedding(self._len_char_vocab, self._char_emb_dim)
        self._fc = tf.keras.layers.Dense(char_emd_dim, activation=tf.nn.relu)

    def call(self, x):
        """

        :param input: Tensor of size [batch_size, max_seq_length, max_word_length]
        :return:
        """

        embeddings = self._char_embed_layer(x) # [batch_size, max_seq_length, max_word_length, char_emd_dim]
        batch_size, max_seq_length, max_word_length, char_emd_dim = tf.shape(embeddings)

        mask = create_padding_mask(x)
        mask = tf.reshape(mask, shape=[-1, 1, 1, max_word_length])

        embeddings = tf.reshape(embeddings, shape=[batch_size * max_seq_length, max_word_length, char_emd_dim])
        encoded = self._encoder(embeddings, mask, True)
        encoded = tf.reshape(encoded, shape=[batch_size, max_seq_length, max_word_length, char_emd_dim])
        encoded = tf.math.reduce_max(encoded, axis=-2)
        encoded = self._fc(encoded)

        return encoded


