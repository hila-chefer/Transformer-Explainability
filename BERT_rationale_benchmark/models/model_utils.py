from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np
from gensim.models import KeyedVectors

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, PackedSequence, pack_padded_sequence, pad_packed_sequence


@dataclass(eq=True, frozen=True)
class PaddedSequence:
    """A utility class for padding variable length sequences mean for RNN input
    This class is in the style of PackedSequence from the PyTorch RNN Utils,
    but is somewhat more manual in approach. It provides the ability to generate masks
    for outputs of the same input dimensions.
    The constructor should never be called directly and should only be called via
    the autopad classmethod.

    We'd love to delete this, but we pad_sequence, pack_padded_sequence, and
    pad_packed_sequence all require shuffling around tuples of information, and some
    convenience methods using these are nice to have.
    """

    data: torch.Tensor
    batch_sizes: torch.Tensor
    batch_first: bool = False

    @classmethod
    def autopad(cls, data, batch_first: bool = False, padding_value=0, device=None) -> 'PaddedSequence':
        # handle tensors of size 0 (single item)
        data_ = []
        for d in data:
            if len(d.size()) == 0:
                d = d.unsqueeze(0)
            data_.append(d)
        padded = pad_sequence(data_, batch_first=batch_first, padding_value=padding_value)
        if batch_first:
            batch_lengths = torch.LongTensor([len(x) for x in data_])
            if any([x == 0 for x in batch_lengths]):
                raise ValueError(
                    "Found a 0 length batch element, this can't possibly be right: {}".format(batch_lengths))
        else:
            # TODO actually test this codepath
            batch_lengths = torch.LongTensor([len(x) for x in data])
        return PaddedSequence(padded, batch_lengths, batch_first).to(device=device)

    def pack_other(self, data: torch.Tensor):
        return pack_padded_sequence(data, self.batch_sizes, batch_first=self.batch_first, enforce_sorted=False)

    @classmethod
    def from_packed_sequence(cls, ps: PackedSequence, batch_first: bool, padding_value=0) -> 'PaddedSequence':
        padded, batch_sizes = pad_packed_sequence(ps, batch_first, padding_value)
        return PaddedSequence(padded, batch_sizes, batch_first)

    def cuda(self) -> 'PaddedSequence':
        return PaddedSequence(self.data.cuda(), self.batch_sizes.cuda(), batch_first=self.batch_first)

    def to(self, dtype=None, device=None, copy=False, non_blocking=False) -> 'PaddedSequence':
        # TODO make to() support all of the torch.Tensor to() variants
        return PaddedSequence(
            self.data.to(dtype=dtype, device=device, copy=copy, non_blocking=non_blocking),
            self.batch_sizes.to(device=device, copy=copy, non_blocking=non_blocking),
            batch_first=self.batch_first)

    def mask(self, on=int(0), off=int(0), device='cpu', size=None, dtype=None) -> torch.Tensor:
        if size is None:
            size = self.data.size()
        out_tensor = torch.zeros(*size, dtype=dtype)
        # TODO this can be done more efficiently
        out_tensor.fill_(off)
        # note to self: these are probably less efficient than explicilty populating the off values instead of the on values.
        if self.batch_first:
            for i, bl in enumerate(self.batch_sizes):
                out_tensor[i, :bl] = on
        else:
            for i, bl in enumerate(self.batch_sizes):
                out_tensor[:bl, i] = on
        return out_tensor.to(device)

    def unpad(self, other: torch.Tensor) -> List[torch.Tensor]:
        out = []
        for o, bl in zip(other, self.batch_sizes):
            out.append(o[:bl])
        return out

    def flip(self) -> 'PaddedSequence':
        return PaddedSequence(self.data.transpose(0, 1), not self.batch_first, self.padding_value)


def extract_embeddings(vocab: Set[str], embedding_file: str, unk_token: str = 'UNK', pad_token: str = 'PAD') -> (
nn.Embedding, Dict[str, int], List[str]):
    vocab = vocab | set([unk_token, pad_token])
    if embedding_file.endswith('.bin'):
        WVs = KeyedVectors.load_word2vec_format(embedding_file, binary=True)

        word_to_vector = dict()
        WV_matrix = np.matrix([WVs[v] for v in WVs.vocab.keys()])

        if unk_token not in WVs:
            mean_vector = np.mean(WV_matrix, axis=0)
            word_to_vector[unk_token] = mean_vector
        if pad_token not in WVs:
            word_to_vector[pad_token] = np.zeros(WVs.vector_size)

        for v in vocab:
            if v in WVs:
                word_to_vector[v] = WVs[v]

        interner = dict()
        deinterner = list()
        vectors = []
        count = 0
        for word in [pad_token, unk_token] + sorted(list(word_to_vector.keys() - {unk_token, pad_token})):
            vector = word_to_vector[word]
            vectors.append(np.array(vector))
            interner[word] = count
            deinterner.append(word)
            count += 1
        vectors = torch.FloatTensor(np.array(vectors))
        embedding = nn.Embedding.from_pretrained(vectors, padding_idx=interner[pad_token])
        embedding.weight.requires_grad = False
        return embedding, interner, deinterner
    elif embedding_file.endswith('.txt'):
        word_to_vector = dict()
        vector = []
        with open(embedding_file, 'r') as inf:
            for line in inf:
                contents = line.strip().split()
                word = contents[0]
                vector = torch.tensor([float(v) for v in contents[1:]]).unsqueeze(0)
                word_to_vector[word] = vector
        embed_size = vector.size()
        if unk_token not in word_to_vector:
            mean_vector = torch.cat(list(word_to_vector.values()), dim=0).mean(dim=0)
            word_to_vector[unk_token] = mean_vector.unsqueeze(0)
        if pad_token not in word_to_vector:
            word_to_vector[pad_token] = torch.zeros(embed_size)
        interner = dict()
        deinterner = list()
        vectors = []
        count = 0
        for word in [pad_token, unk_token] + sorted(list(word_to_vector.keys() - {unk_token, pad_token})):
            vector = word_to_vector[word]
            vectors.append(vector)
            interner[word] = count
            deinterner.append(word)
            count += 1
        vectors = torch.cat(vectors, dim=0)
        embedding = nn.Embedding.from_pretrained(vectors, padding_idx=interner[pad_token])
        embedding.weight.requires_grad = False
        return embedding, interner, deinterner
    else:
        raise ValueError("Unable to open embeddings file {}".format(embedding_file))
