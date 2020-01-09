import numpy as np
import tensorflow as tf
import torch

from tener.misc.pretty_print import print_info
from tener.models.attention.multihead_relative_attn import RelativeMultiHeadAttn, RelativeMultiHeadAttnTorch
from tener.models.attention.multihead_naive_attn import MultiHeadAttention
from tener.models.model_utils import create_padding_mask

tf.random.set_seed(42)
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class TestMultiHeadAttention(tf.test.TestCase):
    BATCH_SIZE = 3
    MAX_SEQ_LENGTH = 10
    D_MODEL = 8


    values =      list([1,4,5,9,3,2,0,0,0,0, 1,3,6,7,7,9,4,2,0,0, 1,3,4,5,6,7,2,0,0,0])
    mask_values = list([1,1,1,1,1,1,0,0,0,0, 1,1,1,1,1,1,1,1,0,0, 1,1,1,1,1,1,1,0,0,0])

    r_w_bias = [[-0.3785, 0.4951, -0.9394, -0.8055],
                [-0.1378, -0.2916, -1.4290, -0.5379]]
    r_r_bias = [[-0.0771, 0.1972, -0.0413, -0.0525],
                [-0.7677, -0.3133, 0.3159, 0.3713]]

    data = np.reshape(values, (BATCH_SIZE, MAX_SEQ_LENGTH))
    data = tf.convert_to_tensor(data)
    embedding = tf.keras.layers.Embedding(10, D_MODEL)
    data = embedding(data)
    X = tf.convert_to_tensor(data, dtype=tf.float32)
    X_TORCH = torch.tensor(data.numpy(), dtype=torch.double)

    MASK = create_padding_mask(tf.convert_to_tensor(np.reshape(mask_values, (BATCH_SIZE, MAX_SEQ_LENGTH))))
    MASK_TORCH = torch.tensor(MASK.numpy(), dtype=torch.double)



    def test_naive_multihead(self):
        num_heads = 2
        attn = MultiHeadAttention(d_model=self.D_MODEL, num_heads=num_heads)
        output, attention_weights = attn(self.X, self.X, self.X, self.MASK)

        # print_info("MultiHeadAttention shape {} {}".format(output.shape, attention_weights.shape))
        assert output.shape == (self.BATCH_SIZE, self.MAX_SEQ_LENGTH, self.D_MODEL)
        assert attention_weights.shape ==(self.BATCH_SIZE, num_heads, self.MAX_SEQ_LENGTH, self.MAX_SEQ_LENGTH)

    def test_relative_multihead(self):
        num_heads = 2
        attn = RelativeMultiHeadAttn(d_model=self.D_MODEL, n_head=num_heads, dropout=0.5,
                                     r_w_bias=tf.Variable(tf.convert_to_tensor(self.r_w_bias)),
                                     r_r_bias=tf.Variable(tf.convert_to_tensor(self.r_r_bias)))
        output = attn(self.X, self.MASK)

        # print_info("RelativeMultiHeadAttn {}".format(output))
        # print_info("RelativeMultiHeadAttn shape {}".format(output.shape))
        assert output.shape == (self.BATCH_SIZE, self.MAX_SEQ_LENGTH, self.D_MODEL)

    def test_relative_multihead_torch(self):
        attn = RelativeMultiHeadAttnTorch(d_model=8, n_head=2, dropout=0.5,
                                          r_w_bias=torch.nn.Parameter(torch.tensor(self.r_w_bias)),
                                          r_r_bias=torch.nn.Parameter(torch.tensor(self.r_r_bias)))
        output = attn(self.X_TORCH.float(), self.MASK_TORCH.reshape((self.BATCH_SIZE, self.MAX_SEQ_LENGTH)))

        assert output.shape == (self.BATCH_SIZE, self.MAX_SEQ_LENGTH, self.D_MODEL)

        # print_info("RelativeMultiHeadAttnTorch {}".format(output))
        # print_info("RelativeMultiHeadAttnTorch shape {}".format(output.shape))


    def test_shift(self):
        attn = RelativeMultiHeadAttn(d_model=16, n_head=4, dropout=0.5, batch_size=2)
        attn_torch = RelativeMultiHeadAttnTorch(d_model=16, n_head=4, dropout=0.5)

        values = list(range(2*2*4*8))
        np_arr = np.reshape(values, (2, 2, 4, 8))
        arr = tf.convert_to_tensor(np_arr, dtype=tf.float32)
        res = attn._shift(arr)
        res = res.numpy()
        arr = torch.tensor(np_arr)
        res1 = attn_torch._shift(arr)
        res1 = res1.numpy()

        assert np.array_equal(res, res1)

    def test_tf_torch_attn(self):
        """
        This was done to check how different is TF and Torch calculations. All in the order of e-2 difference
        :return:
        """
        attn = RelativeMultiHeadAttn(d_model=8, n_head=2, dropout=0.5,
                                     r_w_bias=tf.Variable(tf.convert_to_tensor(self.r_w_bias)),
                                     r_r_bias=tf.Variable(tf.convert_to_tensor(self.r_r_bias)))
        output1 = attn(self.X, self.MASK)

        attn = RelativeMultiHeadAttnTorch(d_model=8, n_head=2, dropout=0.5,
                                          r_w_bias=torch.nn.Parameter(torch.tensor(self.r_w_bias)),
                                          r_r_bias=torch.nn.Parameter(torch.tensor(self.r_r_bias)))
        output2 = attn(self.X_TORCH.float(), self.MASK_TORCH.reshape((self.BATCH_SIZE, self.MAX_SEQ_LENGTH)))

        is_close = np.isclose(output1.numpy(), output2.detach().numpy(), rtol=1e-05, atol=0.0)
        print_info(output1.numpy() - output2.detach().numpy())
        # TODO how to check for the value closeness
        print_info(is_close)

        assert np.all(is_close)
