#! /usr/bin/env python
# coding=utf-8

# /************************************************************************************
# ***
# ***	File Author: Dell, 2018年 09月 10日 星期一 21:27:08 CST
# ***
# ************************************************************************************/

import sys
import os
import argparse

import torch

import data
import model
from torch.autograd import Variable
from torch.nn import functional as F


def parse_arguments():
    p = argparse.ArgumentParser(description='Evaluating Translation Model')
    p.add_argument(
        '-batch_size',
        type=int,
        default=64,
        help='number of epochs for train, [64]')
    p.add_argument(
        '-model',
        type=str,
        default=model.DEFAULT_MODEL,
        help='pre-trained model [' + model.DEFAULT_MODEL + ']')
    p.add_argument(
        '-datafile',
        type=str,
        default="data/en-zh.txt",
        help='Evaluating data file [/data/en-zh.txt]')

    return p.parse_args()


def eval_model(val_iter, model, vocab_size, word_pad):
    model.eval()
    total_loss = 0
    for b, batch in enumerate(val_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src, trg = src.cuda(), trg.cuda()

        output = model(src, trg, teacher_forcing_ratio=0.0)
        loss = F.nll_loss(
            output[1:].view(-1, vocab_size),
            trg[1:].contiguous().view(-1),
            ignore_index=word_pad)
        total_loss += loss.data.item()
    return total_loss


if __name__ == '__main__':
    args = parse_arguments()
    hidden_size = 512
    embed_size = 256

    assert torch.cuda.is_available()

    if not os.path.exists(args.model):
        print("model %s not exists." % (args.model))
        sys.exit(-1)

    valdata, en_field, zh_field = data.translate_dataloader(
        args.datafile, args.batch_size, shuffle=False)
    en_field.vocab = data.load_vocab("models/english.vocab")
    zh_field.vocab = data.load_vocab("models/chinese.vocab")

    en_size = len(en_field.vocab)
    zh_size = len(zh_field.vocab)
    zh_pad = zh_field.vocab.stoi['<pad>']

    seq2seq = torch.load(args.model)
    seq2seq = seq2seq.cuda()

    loss = eval_model(valdata, seq2seq, zh_size, zh_pad)
    print("Evalating loss: %10.4f" % loss)
