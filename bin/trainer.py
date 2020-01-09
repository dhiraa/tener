#!/usr/bin/env python
import datetime

from tensorflow import TensorShape

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


class KerasDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, tf_dataset):
        self._tf_dataset = tf_dataset

        num_batches = 0
        for _ in self._tf_dataset:
            num_batches += 1
        self._num_batches = num_batches

        self.it = iter(self._tf_dataset)

    def __len__(self):
        print_error(self._num_batches)
        return self._num_batches

    def __getitem__(self, item):
        data = next(self.it)
        print_error(data)
        return data

    def on_epoch_end(self):
        self.it = iter(self._tf_dataset)

@gin.configurable
class Trainer:
    def __init__(self,
                 dataset_name=None,
                 model_name=None,
                 epochs=100,
                 checkpoint_path=None):

        # tf.summary.trace_on(graph=True, profiler=True)

        self._dataset = None
        self._model = None
        self._epochs = epochs
        if checkpoint_path is None:
            checkpoint_path = "store/checkpoints/" + model_name

        self._checkpoint_path = checkpoint_path

        if dataset_name == "conll2003":
            self._dataset: CoNLLDataset = CoNLLDataset()

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


    def train(self):

        train_loss = self._model._train_loss
        train_accuracy = self._model._train_accuracy

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = self._checkpoint_path + current_time+ '/train_logs'
        # test_log_dir = self._checkpoint_path + current_time + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        # test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        step = 0


        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # training_generator = KerasDataGenerator(self._dataset.train_dataset)
        # model =  self._model._transformer
        #
        # model.compile()
        # model.fit_generator(generator=training_generator)
        # model.build([TensorShape([self._dataset._batch_size, self._dataset._max_seq_length]),
        #              TensorShape([self._dataset._batch_size,
        #                           self._dataset._max_seq_length,
        #                           self._dataset._max_word_length])])
        #
        #
        #
        #
        # exit()

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # https://github.com/tensorflow/datasets/issues/561
        # https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
        # https://www.dlology.com/blog/an-easy-guide-to-build-new-tensorflow-datasets-and-estimator-with-keras-model/
        # https://www.pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing/
        for epoch in tqdm(range(1, self._epochs+1), desc="Epoch"):

            start = time.time()

            train_loss.reset_states()
            train_accuracy.reset_states()

            for (batch, (words, chars, tar)) in tqdm(enumerate(self._dataset.train_dataset), desc="Batch"):
                self._model.train_step([words, chars],
                                       tar,
                                       text_tokenizer=self._dataset.text_tokenizer,
                                       tag_tokenizer=self._dataset.tags_tokenizer,
                                       is_training=True,
                                       is_log=batch%100 == 0)

                # with train_summary_writer.as_default():
                #     tf.summary.trace_export(
                #         name="tener_trace",
                #         step=step,
                #         profiler_outdir=train_log_dir)

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


