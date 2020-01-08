# import tensorflow as tf
# import time
#
# from config.config import *
# from tener.models.vanialla_transformer import Transformer
# # from datasets import *
#
# from src import CoNLLDataset
# import matplotlib.pyplot as plt
#
#
# def create_padding_mask(seq):
#     seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
#
#     # add extra dimensions to add the padding
#     # to the attention logits.
#     return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
#
# def create_look_ahead_mask(size):
#   mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
#   return mask  # (seq_len, seq_len)
#
#
# dataset = CoNLLDataset()
# dataset.prepare()
#
#
# train_dataset = dataset.train_dataset
# val_dataset = dataset.val_dataset
#
# input_vocab_size = dataset.input_vocab_size
# target_vocab_size = dataset.input_vocab_size
#
# tokenizer_pt = dataset.text_encoder
# tokenizer_en = dataset.tags_encoder
#
# print(train_dataset)
# print(val_dataset)
#
# # for i, j  in train_dataset:
# #     print("text", i.shape)
# #     print("tags", j.shape)
# #     # exit()
#
# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, d_model, warmup_steps=4000):
#         super(CustomSchedule, self).__init__()
#
#         self.d_model = d_model
#         self.d_model = tf.cast(self.d_model, tf.float32)
#
#         self.warmup_steps = warmup_steps
#
#     def __call__(self, step):
#         arg1 = tf.math.rsqrt(step)
#         arg2 = step * (self.warmup_steps ** -1.5)
#
#         return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
#
#
#
# learning_rate = CustomSchedule(d_model)
#
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
#                                      epsilon=1e-9)
#
#
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#     from_logits=True, reduction='none')
#
#
# def loss_function(real, pred):
#     mask = tf.math.logical_not(tf.math.equal(real, 0))
#     loss_ = loss_object(real, pred)
#
#     mask = tf.cast(mask, dtype=loss_.dtype)
#     loss_ *= mask
#
#     return tf.reduce_mean(loss_)
#
#
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
#
#
# transformer = Transformer(num_layers,
#                           d_model,
#                           num_heads,
#                           dff,
#                           input_vocab_size,
#                           target_vocab_size,
#                           pe_input=input_vocab_size,
#                           pe_target=target_vocab_size,
#                           rate=dropout_rate)
#
#
#
# def create_masks(inp, tar):
#     # Encoder padding mask
#     enc_padding_mask = create_padding_mask(inp)
#
#     # Used in the 2nd attention block in the decoder.
#     # This padding mask is used to mask the encoder outputs.
#     dec_padding_mask = create_padding_mask(inp)
#
#     # Used in the 1st attention block in the decoder.
#     # It is used to pad and mask future tokens in the input received by
#     # the decoder.
#     look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
#     dec_target_padding_mask = create_padding_mask(tar)
#     combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
#
#     return enc_padding_mask, combined_mask, dec_padding_mask
#
#
# checkpoint_path = "checkpoints/train"
#
# ckpt = tf.train.Checkpoint(transformer=transformer,
#                            optimizer=optimizer)
#
# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
#
# # if a checkpoint exists, restore the latest checkpoint.
# if ckpt_manager.latest_checkpoint:
#   ckpt.restore(ckpt_manager.latest_checkpoint)
#   print ('Latest checkpoint restored!!')
#
# # The @tf.function trace-compiles train_step into a TF graph for faster
# # execution. The function specializes to the precise shape of the argument
# # tensors. To avoid re-tracing due to the variable sequence lengths or variable
# # batch sizes (the last batch is smaller), use input_signature to specify
# # more generic shapes.
#
# train_step_signature = [
#     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
#     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
# ]
#
#
# @tf.function(input_signature=train_step_signature)
# def train_step(inp, tar):
#     tar_inp = tar[:, :-1]
#     tar_real = tar[:, 1:]
#
#     print("2. >>>>>>>>>>>>>> {} {}".format(inp.shape, tar.shape))
#
#     enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
#
#     with tf.GradientTape() as tape:
#         predictions, _ = transformer(inp,
#                                      tar_inp,
#                                      True,
#                                      enc_padding_mask,
#                                      combined_mask,
#                                      dec_padding_mask)
#         loss = loss_function(tar_real, predictions)
#         print("!!!!! trainer : {}".format(loss))
#
#     gradients = tape.gradient(loss, transformer.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
#
#     train_loss(loss)
#     train_accuracy(tar_real, predictions)
#
#
# def train():
#     for epoch in range(EPOCHS):
#         print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\n\n")
#         start = time.time()
#
#         train_loss.reset_states()
#         train_accuracy.reset_states()
#
#         # inp -> portuguese, tar -> english
#         for (batch, (inp, tar)) in enumerate(train_dataset):
#             print("1. >>>>>>>>>>>>>> {} {} {}".format(batch, inp.shape, tar.shape))
#             train_step(inp, tar)
#
#             if batch % 50 == 0:
#                 print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
#                     epoch + 1, batch, train_loss.result(), train_accuracy.result()))
#
#         if (epoch + 1) % 5 == 0:
#             ckpt_save_path = ckpt_manager.save()
#             print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
#                                                                 ckpt_save_path))
#
#         print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
#                                                             train_loss.result(),
#                                                             train_accuracy.result()))
#
#         print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
#
#
# def evaluate(inp_sentence):
#     start_token = [tokenizer_pt.vocab_size]
#     end_token = [tokenizer_pt.vocab_size + 1]
#
#     # inp sentence is portuguese, hence adding the start and end token
#     inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
#     encoder_input = tf.expand_dims(inp_sentence, 0)
#
#     # as the target is english, the first word to the transformer should be the
#     # english start token.
#     decoder_input = [tokenizer_en.vocab_size]
#     output = tf.expand_dims(decoder_input, 0)
#
#     for i in range(MAX_LENGTH):
#         enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
#             encoder_input, output)
#
#         # predictions.shape == (batch_size, seq_len, vocab_size)
#         predictions, attention_weights = transformer(encoder_input,
#                                                      output,
#                                                      False,
#                                                      enc_padding_mask,
#                                                      combined_mask,
#                                                      dec_padding_mask)
#
#         # select the last word from the seq_len dimension
#         predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
#
#         predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
#
#         # return the result if the predicted_id is equal to the end token
#         if predicted_id == tokenizer_en.vocab_size + 1:
#             return tf.squeeze(output, axis=0), attention_weights
#
#         # concatentate the predicted_id to the output which is given to the decoder
#         # as its input.
#         output = tf.concat([output, predicted_id], axis=-1)
#
#     return tf.squeeze(output, axis=0), attention_weights
#
#
# def plot_attention_weights(attention, sentence, result, layer):
#     fig = plt.figure(figsize=(16, 8))
#
#     sentence = tokenizer_pt.encode(sentence)
#
#     attention = tf.squeeze(attention[layer], axis=0)
#
#     for head in range(attention.shape[0]):
#         ax = fig.add_subplot(2, 4, head + 1)
#
#         # plot the attention weights
#         ax.matshow(attention[head][:-1, :], cmap='viridis')
#
#         fontdict = {'fontsize': 10}
#
#         ax.set_xticks(range(len(sentence) + 2))
#         ax.set_yticks(range(len(result)))
#
#         ax.set_ylim(len(result) - 1.5, -0.5)
#
#         ax.set_xticklabels(
#             ['<start>'] + [tokenizer_pt.decode([i]) for i in sentence] + ['<end>'],
#             fontdict=fontdict, rotation=90)
#
#         ax.set_yticklabels([tokenizer_en.decode([i]) for i in result
#                             if i < tokenizer_en.vocab_size],
#                            fontdict=fontdict)
#
#         ax.set_xlabel('Head {}'.format(head + 1))
#
#     plt.tight_layout()
#     plt.show()
#
# def translate(sentence, plot=None):
#     result, attention_weights = evaluate(sentence)
#
#     predicted_sentence = tokenizer_en.decode([i for i in result
#                                               if i < tokenizer_en.vocab_size])
#
#     print('Input: {}'.format(sentence))
#     print('Predicted translation: {}'.format(predicted_sentence))
#
#     if plot:
#         plot_attention_weights(attention_weights, sentence, result, plot)
#
#
# if __name__ == "__main__":
#     train()
#     # translate("este é um problema que temos que resolver.")
#     # print("Real translation: this is a problem we have to solve .")