from tener.datasets.conll_dataset import CoNLLDataset
from tener.misc.pretty_print import print_info


def test_conll_dataset():
    max_seq_length = 10
    max_char_length = 8
    batch_size = 1
    dataset = CoNLLDataset(in_data_dir="data/test_conll/",
                           out_data_dir="/tmp/processed/",
                           text_column="0",
                           entity_column="3",
                           max_seq_length=max_seq_length,
                           max_word_length=max_char_length,
                           unknown_word='<UNK>',
                           seperator=" ",  # potential error point depending on the datasets
                           quotechar="^",
                           batch_size=batch_size,
                           buffer_size=2000,
                           clear_data=True)

    expected_text = ["_SOS_ begin one two three four five six Seven Eight . _EOS_",
                     '_SOS_ One Two one Three One Two one Nine <UNK> <UNK> _EOS_']
    expected_tag = ["_SOT_ O B-ORG O B-MISC O O O B-MISC O O _EOT_",
                    "_SOT_ B-PER I-PER B-LOC O B-PER I-PER B-LOC O O O _EOT_"]

    expected_char = [['< b > _ S O S _ < e > <U> <U> <U>',
                      '< b > b e g i n < e > <U> <U> <U>',
                      '< b > o n e < e > <U> <U> <U> <U> <U>',
                      '< b > t w o < e > <U> <U> <U> <U> <U>',
                      '< b > t h r e e < e > <U> <U> <U>',
                      '< b > f o u r < e > <U> <U> <U> <U>',
                      '< b > f i v e < e > <U> <U> <U> <U>',
                      '< b > s i x < e > <U> <U> <U> <U> <U>',
                      '< b > S e v e n < e > <U> <U> <U>',
                      '< b > E i g h t < e > <U> <U> <U>',
                      '< b > . < e > <U> <U> <U> <U> <U> <U> <U>',
                      '< b > _ E O S _ < e > <U> <U> <U>'],
                     ['< b > _ S O S _ < e > <U> <U> <U>',
                      '< b > O n e < e > <U> <U> <U> <U> <U>',
                      '< b > T w o < e > <U> <U> <U> <U> <U>',
                      '< b > o n e < e > <U> <U> <U> <U> <U>',
                      '< b > T h r e e < e > <U> <U> <U>',
                      '< b > O n e < e > <U> <U> <U> <U> <U>',
                      '< b > T w o < e > <U> <U> <U> <U> <U>',
                      '< b > o n e < e > <U> <U> <U> <U> <U>',
                      '< b > N i n e < e > <U> <U> <U> <U>',
                      '< b > < U N K > < e > <U> <U> <U>',
                      '< b > < U N K > < e > <U> <U> <U>',
                      '< b > _ E O S _ < e > <U> <U> <U>']
                    ]

    for batch, (features, label) in enumerate(dataset.train_dataset):
        word_ids = features["word_ids"]
        char_ids = features["char_ids"]
        tag_ids = label

        assert word_ids.numpy().shape == (batch_size, max_seq_length+2) # 2 for begin and end words
        assert char_ids.numpy().shape == (batch_size, max_seq_length+2, max_char_length+6) # 6 for begin and end characters
        assert tag_ids.numpy().shape == (batch_size, max_seq_length+2) # 2 for begin and end words

        # TODO works only for batch size of 1...
        decoded_text = dataset.text_tokenizer.sequences_to_texts(word_ids.numpy())
        assert " ".join(decoded_text) == expected_text[batch]

        decoded_tag = dataset.tags_tokenizer.sequences_to_texts(tag_ids.numpy())
        assert " ".join(decoded_tag) == expected_tag[batch]

        for char_id in char_ids:
            decoded_chars = dataset.char_tokenizer.sequences_to_texts(char_id.numpy())

        assert decoded_chars == expected_char[batch]
