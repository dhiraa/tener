import os
from string import punctuation
import pickle
import shutil
import gin
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tensorflow_datasets as tfds

from tener.misc.pretty_print import print_error, print_info


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _mat_feature(mat):
    # mat = np.cast(mat, tf.int64)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=mat.flatten()))


def get_keras_tokenizer(text_corpus, char_level=False, oov_token="<UNK>"):
    # Here we wanted all the special characters to be part of the vocab, as it is important for NER tagging
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token=oov_token, char_level=char_level, lower=False)
    tokenizer.fit_on_texts(text_corpus)
    return tokenizer


def to_tensor(data):
    return tf.data.Dataset.from_tensor_slices(data)


def _conll_to_csv(txt_file_path, out_dir, unknown_word="<UNK>"):
    '''
    Function to convert CoNLL 2003 data set text files into CSV file for each
    example/statement.
    :param txt_file_path: Input text file path
    :param out_dir: Output directory to store CSV files with columns [0,1,2,3]
    :param unknown_word: For Conll2003 dataset column 3 has NA which can be by default filled with `O`
    :return:
    '''
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print_info("Found files in {}".format(out_dir))
        return

    # Read the text file
    # Few gotchas : 1. There are space in text colum or NER column 2. There are empty or NA in NER column
    df = pd.read_csv(txt_file_path,
                     sep=" ",
                     skip_blank_lines=False,
                     header=None).fillna(unknown_word).replace(r'^\s*$', "<SPACE>", regex=True)

    # Filter out the DOCSTART lines
    df = df[~df[0].str.contains("DOCSTART")]
    current_file = []

    # Here were are considering DOCSTART to DOCSTART line, rather sentences with at least 2 or more words
    for i in tqdm(range(len(df)), desc="conll_to_csv"):
        row = df.values[i]
        if row[0] != unknown_word:
            current_file.append(row)
        else:
            # Consider dumping files with size 2
            if len(current_file) > 2:
                current_file = pd.DataFrame(current_file, columns=["0", "1", "2", "3"])
                # Replace NA -> UNKNOWN_WORD -> O (other)
                current_file["3"] = current_file["3"].replace(unknown_word, "O")
                current_file.to_csv(out_dir + "/{}.csv".format(i), index=False)
                current_file = []


def str_list_to_char_index(text, text_char_tonkenizer, max_seq_len=20, max_char_length=10):
    # text_char_tonkenizer = get_keras_tokenizer(text, char_level=True)

    # split the text by spaces i.e list of list of words
    char_data = [text.split(" ") for text in text]

    char_data_encoded = []

    max_char_length = max_char_length + 6 #account for start and end words _b_ and _e_
    for char_seq in char_data:
        char_seq = ["<b>"+word+"<e>" for word in char_seq]
        # print_error(char_seq)
        # get_keras_tokenizer each sentence
        res = text_char_tonkenizer.texts_to_sequences(char_seq)
        # pad it
        res = tf.keras.preprocessing.sequence.pad_sequences(res, padding="post", maxlen=max_char_length)
        # group it as a batch
        res = np.array(res)
        if max_seq_len < len(char_seq):
            res = res[:, :max_seq_len]
        else:
            res = np.pad(res, ((0, max_seq_len - len(char_seq)), (0, 0)), 'constant', constant_values=(0))
        # print_info(res.shape)
        char_data_encoded.append(res)

    return char_data_encoded#, text_char_tonkenizer

@gin.configurable
class CoNLLDataset(object):
    def __init__(self,
                 in_data_dir="data/conll/",
                 out_data_dir="processed/",
                 text_column="0",
                 entity_column="3",
                 max_seq_length=40,
                 max_word_length=10,
                 unknown_word='<UNK>',
                 seperator=" ",  # potential error point depending on the datasets
                 quotechar="^",
                 batch_size=64,
                 buffer_size=2000,
                 clear_data=False):
        """

        :param in_data_dir:
        :param out_data_dir:
        :param text_column:
        :param entity_column:
        :param max_seq_length:
        :param unknown_word:
        :param seperator:
        :param quotechar:
        :param batch_size:
        :param buffer_size:
        """

        if clear_data:
            shutil.rmtree(out_data_dir)

        self._in_data_dir = in_data_dir
        self._out_data_dir = out_data_dir

        self._out_train_dir = out_data_dir + "/train/"
        self._out_test_dir = out_data_dir + "/test/"
        self._out_val_dir = out_data_dir + "/val/"

        self._text_column = text_column
        self._entity_column = entity_column
        self._unknown_word = unknown_word
        self._seperator = seperator
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        # account in for start & stop words
        self._max_seq_length = max_seq_length
        self._max_word_length = max_word_length

        self._start_word = "_SOS_"  # StartOfSentence
        self._end_word = "_EOS_"  # EndOFSentence

        self._start_tag= "_SOT_"  # StartOfSentence
        self._end_tag = "_EOT_"  # EndOFSentence

        self.input_vocab_size = -1
        self.target_vocab_size = -1
        self.text_tokenizer = None
        self.tags_tokenizer = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.prepare()

    def txt_to_csv(self):
        """
        Converts the CoNLL2003 text files into CSV files
        :return:
        """
        _conll_to_csv(txt_file_path=self._in_data_dir+"/train.txt",
                      out_dir=self._out_train_dir)
        _conll_to_csv(txt_file_path=self._in_data_dir+"/test.txt",
                      out_dir=self._out_test_dir)
        _conll_to_csv(txt_file_path=self._in_data_dir+"/val.txt",
                      out_dir=self._out_val_dir)

    def csv_to_data(self, csv_files_path):
        """
        Read text and tags from CSV files
        :param csv_files_path:
        :return: List of text, List of tags
        """
        sentence_feature = []
        tag_label = []

        for df_file in tqdm(os.listdir(csv_files_path), desc="csv_to_data"):

            df_file = os.path.join(csv_files_path, df_file)

            if df_file.endswith(".csv"):
                df = pd.read_csv(df_file)
            elif df_file.endswith(".json"):
                df = pd.read_json(df_file)
            else:
                continue

            df[self._text_column] = df[self._text_column].replace(r'^\s*$', "<SPACE>", regex=True)
            df[self._entity_column] = df[self._entity_column].fillna("O")

            list_of_words = df[self._text_column].astype(str).values.tolist()[:self._max_seq_length]
            list_of_tag = df[self._entity_column].astype(str).values.tolist()[:self._max_seq_length]

            # If the sequence length is less than given max sequence length
            if len(list_of_words) < self._max_seq_length:
                list_of_words = list_of_words + ["<UNK>"] * (self._max_seq_length - len(list_of_words))
                list_of_tag = list_of_tag + ["O"] * (self._max_seq_length - len(list_of_tag))

            assert len(list_of_words) == self._max_seq_length

            list_of_words = [self._start_word] + list_of_words + [self._end_word]
            list_of_tag = [self._start_tag] + list_of_tag + [self._end_tag]

            assert len(list_of_words) == len(list_of_tag)

            sentence = "{}".format(self._seperator).join(list_of_words)
            sentence_feature.append(sentence)
            tag = "{}".format(self._seperator).join(list_of_tag)
            tag_label.append(tag)

        return sentence_feature, tag_label

    def _get_features(self, sentence, char_ids, ner_tags):
        """
        Given different features matrices, this routine wraps the matrices as TF features
        """
        data = {"word_ids": _mat_feature(sentence), "char_ids": _mat_feature(char_ids), "tag_ids": _mat_feature(ner_tags)}
        return data

    def write_tf_records(self, word_ids_padded, tag_ids_padded, char_ids_padded, file_path_name):
        """

        :param sentences: List of text
        :param tags: List of tags
        :param file_path_name:
        :return:
        """
        with tf.io.TFRecordWriter(file_path_name) as writer:
            for word_ids, char_ids, tag_ids in tqdm(zip(word_ids_padded, char_ids_padded, tag_ids_padded),
                                                    desc="write_tf_records : "):
                features = tf.train.Features(feature=self._get_features(word_ids, char_ids, tag_ids))
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())

    def load_or_create_dataset(self, csv_files_path):
        """

        :param csv_files_path:
        :param out_file_path:
        :return:
        """

        print_info("Processing data in {}".format(csv_files_path))
        text_tokenizer_file = self._out_train_dir + "/text_tokenizer.dat"
        char_tokenizer_file = self._out_train_dir + "/char_tokenizer.dat"
        tags_tokenizer_file = self._out_train_dir + "/tags_tokenizer.dat"
        out_file_path = csv_files_path+"/tf_data.tfrecords"

        # Load TFRecords file if found
        if os.path.exists(out_file_path):

            self.text_tokenizer = pickle.load(open(text_tokenizer_file, "rb"))
            self.tags_tokenizer = pickle.load(open(tags_tokenizer_file, "rb"))
            self.char_tokenizer = pickle.load(open(char_tokenizer_file, "rb"))

            dataset = tf.data.TFRecordDataset(out_file_path)
            print_info("Found file {}".format(out_file_path))
            return dataset

        sentences, tags = self.csv_to_data(csv_files_path)

        # Check for pickled tokenizer in the output path and load if available
        if os.path.exists(text_tokenizer_file) and \
                os.path.exists(tags_tokenizer_file):
            print_info("Loading tokenizers...")
            self.text_tokenizer = pickle.load(open(text_tokenizer_file, "rb"))
            self.tags_tokenizer = pickle.load(open(tags_tokenizer_file, "rb"))
            self.char_tokenizer = pickle.load(open(char_tokenizer_file, "rb"))
        else:
            self.char_tokenizer = get_keras_tokenizer(sentences, char_level=True, oov_token="<U>")
            self.text_tokenizer = get_keras_tokenizer(sentences)
            self.tags_tokenizer = get_keras_tokenizer(tags)

            pickle.dump(self.char_tokenizer, open(char_tokenizer_file, "wb"))
            pickle.dump(self.text_tokenizer, open(text_tokenizer_file, "wb"))
            pickle.dump(self.tags_tokenizer, open(tags_tokenizer_file, "wb"))

        char_ids_padded = str_list_to_char_index(text=sentences,
                                                 max_seq_len=self._max_seq_length + 2,
                                                 max_char_length=self._max_word_length,
                                                 text_char_tonkenizer=self.char_tokenizer)

        word_ids = self.text_tokenizer.texts_to_sequences(sentences)
        word_ids_padded = tf.keras.preprocessing.sequence.pad_sequences(word_ids, padding='post')

        tag_ids = self.tags_tokenizer.texts_to_sequences(tags)
        tag_ids_padded = tf.keras.preprocessing.sequence.pad_sequences(tag_ids, padding='post')

        self.write_tf_records(file_path_name=out_file_path,
                              word_ids_padded=word_ids_padded,
                              tag_ids_padded=tag_ids_padded,
                              char_ids_padded=char_ids_padded)

        # Read it back after writing
        print_info("Found file {}".format(out_file_path))
        dataset = tf.data.TFRecordDataset(out_file_path)

        # print_info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print_info(sentences)
        # print_info(tags)
        # print_info(char_ids_padded)
        # print_info(word_ids_padded)
        # print_info(tag_ids_padded)
        # print_info("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

        return dataset

    def data_to_dataset(self):
        self.train_dataset = self.load_or_create_dataset(self._out_train_dir)
        self.val_dataset = self.load_or_create_dataset(self._out_val_dir)
        self.test_dataset = self.load_or_create_dataset(self._out_test_dir)

    def decode(self, serialized_example):
        # 1. define a parser

        features = tf.io.parse_single_example(
            serialized_example,
            features={
                'word_ids': tf.io.FixedLenFeature([self._max_seq_length+2], tf.int64),
                # 6 added since we include <b> and <e> as start and end of word in character indexing
                'char_ids': tf.io.FixedLenFeature([self._max_seq_length+2, self._max_word_length+6], tf.int64),
                'tag_ids': tf.io.FixedLenFeature([self._max_seq_length+2], tf.int64),

            })

        text = tf.reshape(
            tf.cast(features['word_ids'], tf.int64), shape=[self._max_seq_length+2])
        char = tf.reshape(
            tf.cast(features['char_ids'], tf.int64), shape=[self._max_seq_length+2, self._max_word_length+6])
        tags = tf.reshape(
            tf.cast(features['tag_ids'], tf.int64), shape=[self._max_seq_length+2])

        return {"word_ids": text, "char_ids": char}, tags

    def filter_max_length(self, x, y):
        return tf.logical_and(tf.size(x) <= self._max_seq_length,
                              tf.size(y) <= self._max_seq_length)

    def prepare(self):

        print_info("Preparing the Conll2003 dataset...")

        self.txt_to_csv()
        print_info("Loading Tensorflow Dataset...")
        self.data_to_dataset()
        print_info("Done loading Tensorflow Dataset...")


        # Add OutOfVocab to the vocab size
        self.input_vocab_size = len(self.text_tokenizer.word_index) + 1
        self.target_vocab_size = len(self.tags_tokenizer.word_index) + 1

        print_info("Input vocab size is {}".format(self.input_vocab_size))
        print_info("Input tag size is {}".format(self.target_vocab_size))


        self.train_dataset = self.train_dataset.map(map_func=self.decode,
                                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # We have handled this while preparing the dataset
        # self.train_dataset = self.train_dataset.filter(self.filter_max_length)

        # cache the datasets to memory to get a speedup while reading from it.
        self.train_dataset = self.train_dataset.cache()

        # Again we have handled this while preparing the dataset
        # self.train_dataset = self.train_dataset.shuffle(self._buffer_size).padded_batch(
        # self._batch_size, padded_shapes=([-1], [-1]))
        self.train_dataset = self.train_dataset.batch(self._batch_size, drop_remainder=True)
        self.train_dataset = self.train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self.val_dataset = self.val_dataset.map(map_func=self.decode,
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.val_dataset = self.val_dataset.cache()
        self.val_dataset = self.val_dataset.batch(self._batch_size, drop_remainder=True)
        self.val_dataset = self.val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self.test_dataset = self.test_dataset.map(map_func=self.decode,
                                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.test_dataset = self.test_dataset.cache()
        self.test_dataset = self.test_dataset.batch(self._batch_size, drop_remainder=True)
        self.test_dataset = self.test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        print_info("Dataset preparation is over...")





