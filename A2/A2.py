from typing import List
from pprint import pprint
from utils import CorpusReader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def generate_data(words: List[str], window_size: int, k: int, corpus: CorpusReader):
    """ Generate the training data for word2vec skip-gram model
    Args:
        text: the input text
        window_size: the size of the context window
        k: the number of negative samples
        corpus: the corpus object, providing utilities such as word2id, getNegatives, etc.
    """
    # Use for loop and yield
    word_ids = [corpus.word2id[word] for word in words]  # Convert the list of words to a list of word ids

    for center_index, center_id in enumerate(word_ids):
        context_indices = None
        # Iterate over the left context words
        for i in range(max(0, center_index - window_size), center_index):
            context_indices=word_ids[i]
            negative_samples = corpus.getNegatives(center_id, k)
            yield center_id, context_indices, negative_samples

        # Iterate over the right context words
        for i in range(center_index + 1, min(center_index + window_size + 1, len(word_ids))):
            context_indices=word_ids[i]
            negative_samples = corpus.getNegatives(center_id, k)
            yield center_id, context_indices, negative_samples



def batchify(data: List, batch_size: int):
    """ Group a stream into batches and yield them as torch tensors.
    Args:
        data: a list of tuples
        batch_size: the batch size
    Yields:
        a tuple of three torch tensors: center, outside, negative
    """
    assert batch_size < len(data)  # data should be long enough
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        if i > len(data) - batch_size:  # if the last batch is smaller than batch_size, pad it with the first few data
            batch = batch + data[:i + batch_size - len(data)]

        center_batch = []
        outside_batch = []
        negative_batch = []

        for center, outside, negative in batch:
            center_batch.append(center)
            outside_batch.append(outside)
            negative_batch.append(negative)

        center_tensor = torch.tensor(center_batch, dtype=torch.long)
        outside_tensor = torch.tensor(outside_batch, dtype=torch.long)
        negative_tensor = torch.tensor(negative_batch, dtype=torch.long)

        yield center_tensor, outside_tensor, negative_tensor


