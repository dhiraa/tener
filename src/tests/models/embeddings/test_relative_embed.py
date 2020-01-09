import sys

from tener.misc.pretty_print import print_info

sys.path.append("/opt/vlab/tener/src/")
import tensorflow as tf
from tener.models.embeddings.relative_embed import RelativeSinusoidalPositionalEmbedding, RelativeSinusoidalPositionalEmbeddingTorch

import torch

class TestRelativeSinusoidalPositionalEmbedding(tf.test.TestCase):
    def test_relative_sin_tener(self):
        embedding_dim = 16
        embd = RelativeSinusoidalPositionalEmbedding(embedding_dim=embedding_dim, padding_idx=0)
        tensor = torch.tensor([[1, 2, 3, 4], [2, 3, 1, 4], [3, 1, 1, 4]])
        embd = embd(tensor)
        # print_info("TF: RelativeSinusoidalPositionalEmbedding shape: {}".format(embd.shape))
        assert embd.shape == (tensor.shape[1]*2, embedding_dim)

    def test_relative_sin_torch(self):
        embedding_dim = 16
        embd = RelativeSinusoidalPositionalEmbeddingTorch(embedding_dim=embedding_dim, padding_idx=0)
        tensor = torch.tensor([[1, 2, 3], [2, 3, 1]])
        embd = embd(tensor)
        # print_info("Torch: RelativeSinusoidalPositionalEmbeddingTorch shape: {}".format(embd.shape))
        assert embd.shape == (tensor.shape[1]*2, embedding_dim)

