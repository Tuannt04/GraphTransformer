from tensorflow.python.platform import gfile
import re
import os
from collections import Counter

# Các Token đặc biệt
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

def build_vocab(data_path, vocab_path, max_size=16500):
    print("--- Dang xay dung tu dien {0} tu {1} ---".format(vocab_path, data_path))
    counter = Counter()
    if not gfile.Exists(data_path):
        raise ValueError("Khong tim thay file du lieu {0}".format(data_path))

    with gfile.GFile(data_path, mode="r") as f:
        for line in f:
            tokens = line.strip().split()
            counter.update(tokens)
    
    count_pairs = counter.most_common(max_size)
    words = [w[0] for w in count_pairs if w[0] not in _START_VOCAB]
    vocab = _START_VOCAB + words
    
    with gfile.GFile(vocab_path, mode="w") as f:
        for w in vocab:
            f.write(w + "\n")
    print("--- Hoan tat! Vocab co {0} tu ---".format(len(vocab)))

def initialize_vocabulary(vocabulary_path, data_path_for_build=None):
    if not gfile.Exists(vocabulary_path) or os.path.getsize(vocabulary_path) == 0:
        if data_path_for_build:
            build_vocab(data_path_for_build, vocabulary_path)
        else:
            raise ValueError("File {0} khong ton tai va khong co data nguon.".format(vocabulary_path))

    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
        rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab