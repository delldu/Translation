#! /usr/bin/env python
# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, 2018年 09月 21日 星期五 10:25:44 CST
# ***
# ************************************************************************************/

import pickle

from torchtext import data


def english_token(x):
    return [w for w in x.split(" ") if len(w) > 0]


EnglishText = data.Field(sequential=True, tokenize=english_token, lower=True, include_lengths=True)


def chinese_token(x):
    return [w for w in x.split(" ") if len(w) > 0]


ChineseText = data.Field(sequential=True, tokenize=chinese_token, include_lengths=True)


class TranslateDataset(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, src_field, trg_field, sep='\t', **kwargs):
        """Create an dataset instance given a path and fields.
        Arguments:
            path: Path to the data file.
            src_field: The field that will be used for source data.
            trg_field: The field that will be used for destion data.
            kwargs: Passed to the constructor of data.Dataset.
        """

        fields = [('src', src_field), ('trg', trg_field)]
        examples = []
        with open(path, errors='ignore') as f:
            for line in f:
                s = line.strip().split(sep)
                if len(s) != 2:
                    continue

                src, trg = s[0], s[1]
                e = data.Example()
                setattr(e, "src", src_field.preprocess(src))
                setattr(e, "trg", trg_field.preprocess(trg))
                examples.append(e)

        super(TranslateDataset, self).__init__(examples, fields, **kwargs)


def translate_dataloader(datafile, batchsize, shuffle=False):
    src_field = EnglishText
    trg_field = ChineseText

    dataset = TranslateDataset(datafile, src_field, trg_field)
    src_field.build_vocab(dataset)
    trg_field.build_vocab(dataset)

    dataiter = data.Iterator(dataset, batchsize, shuffle, repeat=False)
    # dataiter.init_epoch()

    return dataiter, src_field, trg_field


def save_vocab(vocab, filename):
    with open(filename, 'wb') as f:
        pickle.dump(vocab, f)


def load_vocab(filename):
    with open(filename, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def test():
    x, _, _ = translate_dataloader("data/en-zh.txt", 32, shuffle=False)
    batch = next(iter(x))
    src, trg = batch.src, batch.trg
    print("src: ", type(src), src.size(), src)
    print("trg: ", type(trg), trg.size(), trg)
