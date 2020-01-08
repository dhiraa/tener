__author__ = "Mageswaran Dhandapani"
__copyright__ = "Copyright 2020, The Dhira Project"
__credits__ = []
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Mageswaran Dhandapani"
__email__ = "mageswaran1989@gmail.com"
__status__ = "Developement"

import tensorflow as tf

def set_rng_seed(rng_seed: int = None,
                 random: bool = True,
                 numpy: bool = True,
                 pytorch: bool = True,
                 tensorflow: bool = True,
                 deterministic:bool = True):
    """
    Set the module's random number seed.
    :param rng_seed: Set the random number of these modules to the default one is generated randomly.
    :param random: Whether to set the seed of the random module that comes with python to rng_seed.
    :param numpy: Whether to set numpy seed to rng_seed.
    :param pytorch: Whether to set seed for pytorch to rng_seed (set torch.manual_seed and torch.cuda.manual_seed_all).
    :param tensorflow: Whether to set seed for Tensorflow engine
    :param deterministic: Whether to set torch.backends.cudnn.deterministic to pytorch to True
    :return:
    """
    if rng_seed is None:
        import time
        rng_seed = int(time.time()%1000000)
    if random:
        import random
        random.seed(rng_seed)
    if numpy:
        try:
            import numpy
            numpy.random.seed(rng_seed)
        except:
            pass
    if pytorch:
        try:
            import torch
            torch.manual_seed(rng_seed)
            torch.cuda.manual_seed_all(rng_seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
        except:
            pass
    if tensorflow:
        import tensorflow as tf
        tf.set_randon_seed(rng_seed)
    return rng_seed


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)