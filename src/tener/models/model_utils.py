import tensorflow as tf


def create_padding_mask(seq):
    """

    :param seq: Tensor [batch_size, ]
    :return:
    """
    seq = tf.cast(tf.math.not_equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :] # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def get_sequence_length(sequence_ids, pad_word_id=0):
    '''
    Returns the sequence length, droping out all the padded tokens if the sequence is padded
    :param sequence_ids: Tensor(shape=[batch_size, doc_length])
    :param pad_word_id: 0 is default
    :return: Array of Document lengths of size batch_size
    '''
    flag = tf.greater_equal(sequence_ids, pad_word_id+1)
    flag = tf.cast(flag, tf.int32)
    length = tf.reduce_sum(flag, 1)
    length = tf.cast(length, tf.int32)
    return length