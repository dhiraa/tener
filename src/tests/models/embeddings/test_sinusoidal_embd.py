import sys

from tener.misc.pretty_print import print_info

sys.path.append("/opt/vlab/tener/src/")
from tener.models.embeddings.sinusoidal_embd import SinusoidalPositionalEmbeddingNaive, SinusoidalPositionalEmbedding, SinusoidalPositionalEmbeddingTorch, LearnedPositionalEmbedding
import tensorflow as tf

import torch

# https://medium.com/testcult/intro-to-test-framework-pytest-5b1ce4d011ae
# https://stackoverflow.com/questions/48234032/run-py-test-test-in-different-process


class TestSinusoidalPositionalEmbedding(tf.test.TestCase):
    def test_sin_naive(self):
        embedding_size = 16
        embd = SinusoidalPositionalEmbeddingNaive(maximum_position_encoding=5, d_model=embedding_size)
        tensor = tf.convert_to_tensor([[1, 2, 3], [2, 3, 1]])
        embd = embd(tensor)
        # print_info("T2T: TestSinusoidalPositionalEmbeddingNaive shape: {}".format(embd.shape))
        assert embd.shape == (1, tensor.shape[1], embedding_size)

    def test_sin_tener(self):
        embedding_size = 16
        embd = SinusoidalPositionalEmbedding(embedding_dim=embedding_size, padding_idx=0)
        tensor = tf.convert_to_tensor([[1, 2, 3], [2, 3, 1]])
        embd = embd(tensor)
        # print_info("SinusoidalPositionalEmbedding shape: {}".format(embd.shape))
        assert embd.shape == (tensor.shape[0], tensor.shape[1], embedding_size)

    def test_learned_pos_embd_tener(self):
        embedding_size = 16
        embd = LearnedPositionalEmbedding(embedding_dim=embedding_size, padding_idx=0, num_embeddings=5)
        tensor = tf.convert_to_tensor([[1, 2, 3], [2, 3, 1]])
        embd = embd(tensor)
        # print_info("LearnedPositionalEmbedding shape: {}".format(embd.shape))
        assert embd.shape == (tensor.shape[0], tensor.shape[1], embedding_size)

    def test_sin_torch(self):
        embedding_size = 16
        embd_torch = SinusoidalPositionalEmbeddingTorch(embedding_dim=16, padding_idx=0)
        tensor = torch.tensor([[1, 2, 3], [2, 3, 1]])
        embd_torch = embd_torch(tensor)
        # print_info("Torch: SinusoidalPositionalEmbeddingTorch shape: {}".format(embd_torch.shape))
        assert embd_torch.shape == (tensor.shape[0], tensor.shape[1], embedding_size)
