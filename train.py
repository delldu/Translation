#! /usr/bin/env python
# coding=utf-8

# /************************************************************************************
# ***
# ***   File Author: Dell, 2018年 09月 10日 星期一 21:27:08 CST
# ***
# ************************************************************************************/

import os
import argparse

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F

import data
import model


def parse_arguments():
    p = argparse.ArgumentParser(description='Training Translation Model')
    p.add_argument(
        '-epochs', type=int, default=100, help='number of epochs [100]')
    p.add_argument(
        '-batch_size',
        type=int,
        default=64,
        help='number of epochs for train, [64]')
    p.add_argument(
        '-lr', type=float, default=0.0001, help='learning rate [0.0001]')
    p.add_argument(
        '-model',
        type=str,
        default=model.DEFAULT_MODEL,
        help='pre-trained model [' + model.DEFAULT_MODEL + ']')

    return p.parse_args()


def save_steps(epochs):
    n = int((epochs + 1) / 10)
    if n < 10:
        n = 10
    n = 10 * int((n + 9) / 10)  # round to 10x times
    return n


def save_model(model, steps):
    if not os.path.isdir("models"):
        os.makedirs("models")
    save_path = 'models/model.pt-{}'.format(steps)
    print("Saving " + save_path + " ... ")
    torch.save(model, save_path)


def train_epoch(train_iter, model, optimizer, vocab_size, word_pad):
    model.train()

    grad_clip = 5.0
    loss = 0
    epoch_loss = 0
    for b, batch in enumerate(train_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src, trg = src.cuda(), trg.cuda()
        optimizer.zero_grad()
        output = model(src, trg)
        loss = F.nll_loss(
            output[1:].view(-1, vocab_size),
            trg[1:].contiguous().view(-1),
            ignore_index=word_pad)
        loss.backward()
        clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()
        loss += loss.data.item()
        epoch_loss += loss

        if b % 100 == 0 and b != 0:
            loss = loss / 100
            print("Training batch %d, loss:%8.4f" % (b, loss))
            loss = 0
    return epoch_loss


def train_model(loader, epochs, model, optimizer, vocab_size, word_pad):
    print("Start training ...")

    save_interval = save_steps(epochs)
    for epoch in range(1, epochs + 1):
        loss = train_epoch(loader, model, optimizer, vocab_size, word_pad)
        print("Training epoch %d/%d, loss: %10.4f" % (epoch, epochs, loss))
        if epoch % save_interval == 0:
            save_model(model, epoch)
    print("Training finished.")


if __name__ == '__main__':
    args = parse_arguments()
    hidden_size = 512
    embed_size = 256
    assert torch.cuda.is_available()

    traindata, en_field, zh_field = data.translate_dataloader(
        "data/en-zh.txt", args.batch_size, shuffle=True)
    data.save_vocab(en_field.vocab, "models/english.vocab")
    data.save_vocab(zh_field.vocab, "models/chinese.vocab")

    en_size = len(en_field.vocab)
    zh_size = len(zh_field.vocab)
    zh_pad = zh_field.vocab.stoi['<pad>']

    if os.path.exists(args.model):
        seq2seq = torch.load(args.model)
        seq2seq = seq2seq.cuda()
    else:
        encoder = model.Encoder(
            en_size, embed_size, hidden_size, n_layers=2, dropout=0.5)
        decoder = model.Decoder(
            embed_size, hidden_size, zh_size, n_layers=1, dropout=0.5)
        seq2seq = model.Seq2Seq(encoder, decoder).cuda()

    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    train_model(traindata, args.epochs, seq2seq, optimizer, zh_size, zh_pad)
