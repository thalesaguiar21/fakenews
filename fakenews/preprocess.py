""" Methods for cleaning the FakeNews dataset """
import os
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk


def read_data(fakespath, realspath):
    """ Load the real and fake news into memory """
    fakedata = pd.read_csv(fakespath)
    realdata = pd.read_csv(realspath)
    print('Succesfully read data from:')
    print(f'Fakes: {fakespath}')
    print(f'Reals: {realspath}')
    return fakedata, realdata


def remove_missing(*datasets, chars=None):
    """ Finds and remove samples with nothing the characters given

    Args:
        datasets (list): list of datasets
        chars (list): list of characters to remove, defaults to None
    """
    for data in datasets:
        empty = []
        for i, row in data.iterrows():
            if _is_empty(row.text, chars):
                empty.append(i)
        data.drop(index=empty, inplace=True)


def _is_empty(value, chars):
    return value.strip(chars) == '' or value == np.nan


def remove_publisher(data):
    """ Remove publisher information from a string

    Args:
        data (dataframe): a dataframe with 'text' column

    Example:
        >>> sample = data.text[0]
        >>> sample
        'WASHINGTON DC - Text body'
        >>> remove_publisher(sample)
        'Text body'
    """
    for __, row in data.iterrows():
        text = row['text']
        text_parts = text.strip().split(' - ', maxsplit=1)
        _join_no_publisher(text_parts)
        row['text'] = text_parts[-1]


def _join_no_publisher(text):
    if len(text) == 2 and len(text[0]) > 260:
        text[-1] = text[0] + ' - ' + text[1]


def _merge_title_and_text(data):
    data['text'] = data['title'] + ' ' + data['text']
    data.drop(columns='title', inplace=True)


def _remove_not_text_cols(data):
    cols = ['subject', 'date']
    data.drop(columns=cols, inplace=True)


def _tokenize(data):
    stopwords = set(nltk.corpus.stopwords.words("english"))
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokenized_sentences = []
    for news in data['text'].values:
        tmp = []
        sentences = nltk.sent_tokenize(news)
        for sent in sentences:
            sent = sent.lower()
            tokens = tokenizer.tokenize(sent)
            filtered_words = [w.strip() for w in tokens
                              if _valid_word(w, stopwords)]
            tmp.extend(filtered_words)
        tokenized_sentences.append(tmp)
    return tokenized_sentences


def _valid_word(word, stopwords):
    return word not in stopwords and len(word) > 1


def add_class_col(*datasets):
    """ Labels each dataset sequentially, starting at 1 """
    for i, data in enumerate(datasets):
        data['cls'] = np.ones(len(data.index), dtype=np.int) * i


def truncate_news(data, maxlen=None):
    """ Truncate every new to a given length (num of words).

    Args:
        data: the tokenizes data
        maxlen: where to truncate, defaults to None.
            When None, it will truncate to avg + 3*std of the data
    """
    news_nwords = [len(sent) for sent in data]
    if maxlen is None:
        maxlen = np.mean(news_nwords) + 3*np.std(news_nwords)
        maxlen = math.ceil(maxlen)
    print(f'Truncating at {maxlen}')
    for idx, new in enumerate(data):
        data[idx] = new[:maxlen]


def plot_news_sizes(news, **kwargs):
    """ Plots a histogram of the number of words in each news """
    plt.hist([len(new) for new in news], bins=500, **kwargs)
    plt.show()
    plt.close()


def run(fakespath, realspath):
    """ Executes every step neceesary to clean the FakeNew dataset
    Args:
        fakespath: absolute path to fake data
        realspath: absolute path to real data

    Return:
        2d-list with tokenized news
    """
    fakenews, realnews = tuple(_fix_paths(fakespath, realspath))
    fakenews, realnews = read_data(fakespath, realspath)
    print('Removing rows without text...')
    remove_missing(realnews, fakenews)
    print('Removing publisher information...')
    remove_publisher(realnews)
    print('Adding class column...')
    add_class_col(realnews, fakenews)
    print('Merging fakes and reals')
    database = pd.concat([fakenews, realnews], ignore_index=True)
    del fakenews, realnews

    print('Merging titles and bodies...')
    _merge_title_and_text(database)
    print('Removing subjects and date...')
    _remove_not_text_cols(database)
    print('Tokenizing data...')
    sentences, labels = _tokenize(database), database['cls']
    return sentences, labels


def _fix_paths(*paths):
    for path in paths:
        base = os.path.dirname(__file__)
        yield os.path.join(base, path)
