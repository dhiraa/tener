#!/usr/bin/env python
import datetime

__author__ = "Mageswaran Dhandapani"
__copyright__ = "Copyright 2020, The Dhira Project"
__credits__ = []
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Mageswaran Dhandapani"
__email__ = "mageswaran1989@gmail.com"
__status__ = "Developement"

"""
"""
import sys
import os
import time

sys.path.append("./src/")
sys.path.append("/opt/vlab/tener/src/")

import gin
from absl import app
from absl import flags
from tqdm import tqdm
import tensorflow as tf

from tener.datasets.conll_dataset import CoNLLDataset
from tener.models.vanialla_transformer import VanillaTransformerModel
from tener.models.tener_transformer import TenerModel

FLAGS = flags.FLAGS
flags.DEFINE_string("config_file", "Google Gin config file format", "/path/to/*.gin file")

from tener.misc.pretty_print import print_info, print_error


@gin.configurable
class Trainer:
    def __init__(self,
                 dataset_name=None,
                 model_name=None,
                 epochs=100,
                 checkpoint_path="store/checkpoints/train/"):
        self._dataset = None
        self._model = None
        self._epochs = epochs
        self._checkpoint_path = checkpoint_path

        if dataset_name == "conll2003":
            self._dataset = CoNLLDataset()

        if model_name == "vanilla_transformer":
            self._model = VanillaTransformerModel(input_vocab_size=self._dataset.input_vocab_size,
                                                  target_vocab_size=self._dataset.target_vocab_size)
        elif model_name == "tener":
            self._model = TenerModel(input_vocab_size=self._dataset.input_vocab_size,
                                     target_vocab_size=self._dataset.target_vocab_size)


        self._ckpt_manager = tf.train.CheckpointManager(self._model.ckpt(), checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if self._ckpt_manager.latest_checkpoint:
            self._model.ckpt().restore(self._ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')
        self._word_embedding = tf.keras.layers.Embedding(self._dataset.input_vocab_size, self._model._word_d_model)


    def train(self):

        train_loss = self._model._train_loss
        train_accuracy = self._model._train_accuracy

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = self._checkpoint_path + current_time+ '/train_logs'
        # test_log_dir = self._checkpoint_path + current_time + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        # test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        step = 0

        # self._model._transformer.build({"word_ids": (None, 32, 32, 3), "char_ids": (None, 32, 32, 3)})
        # self._model._transformer.compile(optimizer="adam",  # Optimizer
        #                                  # Loss function to minimize
        #                                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        #                                  # List of metrics to monitor
        #                                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        #
        # # x_train, y_train = self._dataset.train_dataset
        # history = self._model._transformer.fit(self._dataset.train_dataset,
        #                                        batch_size=None,
        #                                        epochs=1)
        #
        # print_info(self._model._transformer.summary())
        #
        # tf.keras.utils.plot_model(
        #     self._model._transformer,
        #     to_file='tener_model.png',
        #     show_shapes=True,
        #     show_layer_names=True,
        #     rankdir='TB',
        #     expand_nested=False,
        #     dpi=96
        # )
        #
        # exit()

        # https://github.com/tensorflow/datasets/issues/561
        # https://www.dlology.com/blog/an-easy-guide-to-build-new-tensorflow-datasets-and-estimator-with-keras-model/
        for epoch in tqdm(range(1, self._epochs+1), desc="Epoch"):

            start = time.time()

            train_loss.reset_states()
            train_accuracy.reset_states()

            for (batch, (inp, tar)) in tqdm(enumerate(self._dataset.train_dataset), desc="Batch"):
                # print_info("1. >>>>>>>>>>>>>> {} {}".format(batch, inp))
                # exit()

                tf.summary.trace_on(graph=True, profiler=True)
                self._model.train_step(inp, tar, is_training=True, is_log=batch%100 == 0)

                with train_summary_writer.as_default():
                    tf.summary.trace_export(
                        name="tener_trace",
                        step=step,
                        profiler_outdir=train_log_dir)

                if batch % 50 == 0:
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', train_loss.result(), step=step)
                        tf.summary.scalar('accuracy', train_accuracy.result(), step=step)
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch, batch, train_loss.result(), train_accuracy.result()))
                step += 1

            if epoch % 5 == 0:
                ckpt_save_path = self._ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch,
                                                                    ckpt_save_path))

            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch,
                                                                train_loss.result(),
                                                                train_accuracy.result()))

            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

            # Reset metrics every epoch
            train_loss.reset_states()
            # test_loss.reset_states()
            train_accuracy.reset_states()
            # test_accuracy.reset_states()


def main(argv):
    gin.parse_config_file(FLAGS.config_file)
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    app.run(main)


