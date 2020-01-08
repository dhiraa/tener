import sys

from tener.misc.pretty_print import print_info

sys.path.append("/opt/vlab/tener/src/")
import tensorflow as tf
from tener.models.embeddings.relative_embed import RelativeSinusoidalPositionalEmbedding, RelativeSinusoidalPositionalEmbeddingTorch

import torch

class TestRelativeSinusoidalPositionalEmbedding(tf.test.TestCase):
    def test_relative_sin_tener(self):
        embd = RelativeSinusoidalPositionalEmbedding(embedding_dim=16, padding_idx=0)
        tensor = torch.tensor([[1, 2, 3], [2, 3, 1]])
        embd = embd(tensor)
        print_info("TF: RelativeSinusoidalPositionalEmbedding shape: {}".format(embd.shape))

    def test_relative_sin_torch(self):
        embd = RelativeSinusoidalPositionalEmbeddingTorch(embedding_dim=16, padding_idx=0)
        tensor = torch.tensor([[1, 2, 3], [2, 3, 1]])
        embd = embd(tensor)
        print_info("Torch: RelativeSinusoidalPositionalEmbeddingTorch shape: {}".format(embd.shape))
