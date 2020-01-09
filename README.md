# Tener [In-progress]()

Tensorflow-Keras port of https://github.com/fastnlp/TENER. 

The motivation behind this port is:
- To check the claim that this architecture can perrform well in small dataset 
- To build the basic data preprocessing from scratch
- Implement the Tener architecture in Tensorflow 
- Tryout different position strategy after mimiking the basic metrics

# Checklist
- [x] Dataset preparation and test cases
- [x] Gin-config
- [x] Plug and play trainer for datasets and model architecture
- [x] Vanilla transformer model integration 
- [x] Tener transformer model 
    - [x] Embeddings
        - [x] Sinusoidal
        - [x] RelativeSinusoidal
        - [x] Character Embedding
    - [x] Attention
        - [x] MultiHeadNaive 
        - [X] MiltiHeadRelative
- [ ] Tuning and Debugging
    - [ ] Vanilla Transformer Model
    - [ ] Tener Transformer Model

# Setup

```
pip install -r requirements.txt
```

# Module Design
- Google [Gin-Config](https://github.com/google/gin-config) based configuration
- Trainer script that reads the config and selects the dataset and model to be used
- Model class that encapsulates the Keras model layers, loss, metrics and train step

```
gin config file ---> trainer ---> dataset and model ---> Keras Model 
```

# How to run?:

- Train
```
cd tener/
# To use vanilla transformer architecture from Tensorflow Tutorial
python bin/trainer.py --config_file=config/vanilla_transformer.gin
# Tener absed architecture 
python bin/trainer.py --config_file=config/tener.gin
```

- Test

```
cd tener/src/

pytest -s

# some times pytest can be picked from global installation
# breaking the pytest to use your environment related packages
/home/{user_home}/anaconda3/envs/{env}/bin/pytest -s

```

# References:
- Transformers: Attention Is All You Need - 2017
    - https://arxiv.org/pdf/1706.03762.pdf
    - Original Code from Google Brain team : https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    - https://datascience.stackexchange.com/questions/51065/what-is-the-positional-encoding-in-the-transformer-model
    - https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/
    - https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/?utm_source=blog&utm_medium=comprehensive-guide-attention-mechanism-deep-learning
    - http://jalammar.github.io/illustrated-transformer/
    - https://datascience.stackexchange.com/questions/51065/what-is-the-positional-encoding-in-the-transformer-model
    - https://github.com/tensorflow/nmt
    - https://www.youtube.com/watch?v=53YvP6gdD7U
    - https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/
    - http://mlexplained.com/2017/12/29/attention-is-all-you-need-explained/
    - https://medium.com/@_init_/how-self-attention-with-relative-position-representations-works-28173b8c245a
    - https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#.XeDjgtHhVXM


- Pytorch:
    - https://github.com/huggingface/transformers

- Tensorflow:
    - https://www.tensorflow.org/tutorials/text/transformer (Basics)


- Information Extraction with Transformer:
    - TENER: Adapting Transformer Encoder for Name Entity Recognition: https://arxiv.org/pdf/1911.04474.pdf
    - Chargrid + Transformer : https://www.groundai.com/project/bertgrid-contextualized-embedding-for-2d-document-representation-and-understanding/1
    - Bert Grid : https://arxiv.org/pdf/1909.04948.pdf