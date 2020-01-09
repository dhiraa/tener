import tensorflow as tf
import tensorflow_addons as tfa
import gin

from tener.misc.pretty_print import print_error, print_info, print_warn
from tener.models.embeddings.character_embd import TransformerCharEncoding
from tener.models.embeddings.sinusoidal_embd import positional_encoding
from tener.models.layers.tener import TenerKerasModel
from tener.utils import CustomSchedule
from tener.models.model_utils import get_sequence_length, create_padding_mask

"""
Transformer modules
- MultiHead Attention
- Encoder Layer
- Decoder Layer
- Encoder
- Decoder
- Transformer
"""

train_step_signature = [
    {"word_ids": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
     "char_ids": tf.TensorSpec(shape=(None, None, None), dtype=tf.int64)},
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=None, dtype=tf.bool),
]


@gin.configurable
class TenerModel(object):
    def __init__(self,
                 input_vocab_size,
                 target_vocab_size,
                 num_layers=4,
                 word_d_model=128,
                 char_d_model=32,
                 num_heads=8,
                 dff=512,
                 use_crf=True,
                 attntype="naive",
                 is_char_embd=True,
                 rate=0.1):
        self._word_d_model = word_d_model

        self._transformer = TenerKerasModel(num_layers=num_layers,
                                            num_heads=num_heads,
                                            dff=dff,
                                            input_vocab_size=input_vocab_size,
                                            target_vocab_size=target_vocab_size,
                                            rate=0.1,
                                            word_d_model=word_d_model,
                                            char_d_model=char_d_model)

        self._train_loss = tf.keras.metrics.Mean(name='train_loss')

        self.use_crf = use_crf
        if not self.use_crf:
            self._train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        else:
            self._train_accuracy = tf.metrics.Accuracy(name="train_accuracy")

    def _loss_naive(self, real, logits, is_training):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, logits)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def _loss(self, real, logits, is_training):
        real = tf.cast(real, tf.int32)
        mask = tf.math.logical_not(tf.math.equal(real, 0))

        sequence_lengths = get_sequence_length(real)
        log_likelihood, trans_params = tfa.text.crf.crf_log_likelihood(inputs=logits,
                                                                       tag_indices=real,
                                                                       sequence_lengths=sequence_lengths)
        mask = tf.cast(mask, dtype=log_likelihood.dtype)
        # log_likelihood *= mask # TODO what happens here?
        loss = tf.reduce_mean(-log_likelihood)

        viterbi_seq, best_score = tfa.text.crf.crf_decode(logits, trans_params, sequence_lengths)
        return viterbi_seq, loss

    def _learning_rate(self):
        return CustomSchedule(self._word_d_model)

    def _optimizer(self):
        learning_rate = self._learning_rate()
        return tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                        epsilon=1e-9)

    def ckpt(self):
        return tf.train.Checkpoint(transformer=self._transformer,
                                   optimizer=self._optimizer())

    # @tf.function(input_signature=train_step_signature)
    def train_step(self, inp, tar, is_training=True, is_log=False, text_tokenizer=None, tag_tokenizer=None):
        """

        :param inp: Dict {"word_ids" : Tensor[batch_size, max_seq_length] ,
                          "char_ids" : Tensor[batch_size, max_seq_length, max_char_length]}
        :param tar: Tensor[batch_size, max_seq_length]
        :param is_training: True/False
        :return: None
        """
        with tf.GradientTape() as tape:
            logits = self._transformer(inp)
            if self.use_crf:
                predictions, loss = self._loss(logits=logits, real=tar, is_training=is_training)
            else:
                loss = self._loss_naive(real=tar, logits=logits, is_training=is_training)
                predictions = logits

        gradients = tape.gradient(loss, self._transformer.trainable_variables)
        self._optimizer().apply_gradients(zip(gradients, self._transformer.trainable_variables))

        self._train_loss(loss)
        self._train_accuracy(tar, predictions)

        if is_log:
            # print_info("Input : {} {}".format(inp["word_ids"], inp["word_ids"].shape))
            # print_info("Input : {} {}".format(inp[0], inp[1].shape))
            # print_info("Target : {}".format(tar))
            # print_info("Predictions : {}".format(predictions))
            print_error(tag_tokenizer.word_index)
            predictions = tf.argmax(predictions, axis=-1)
            if text_tokenizer and tag_tokenizer:
                texts = text_tokenizer.sequences_to_texts(inp[0].numpy())
                actual_tags = tag_tokenizer.sequences_to_texts(tar.numpy())
                pred_tags = tag_tokenizer.sequences_to_texts(predictions.numpy())

                for text, actual_tag, pred_tag, pred_id in zip(texts, actual_tags, pred_tags, predictions.numpy()):
                    print_info("Text: {}".format(text))
                    print_warn("ATag: {}".format(actual_tag))
                    print_warn("PTag: {}".format(pred_tag))
                    print_warn("PTag: {}".format(pred_id))
                    print("\n")






