from typing import List, Tuple

import numpy as np
import spacy
from scipy.spatial.distance import cdist
from spacy.parts_of_speech import NOUN, ADJ, VERB, ADV
from spacy.tokens import Doc, Token

from data.db import WordCluster, Word

try:
    import cupy
    using_gpu = True
except ModuleNotFoundError:
    using_gpu = False


def reduce_to_first(embeddings: np.ndarray):
    return embeddings[0]


def reduce_by_sum(embeddings: np.ndarray):
    return embeddings.sum(0)


def reduce_by_avg(embeddings: np.ndarray):
    return embeddings.mean(0)


reduction_function = {
    'first': reduce_to_first,
    'sum': reduce_by_sum,
    'avg': reduce_by_avg
}


class EmbeddingExtractor:
    def __init__(self, embedding_reducer=reduce_to_first,
                 target_pos_set=frozenset({NOUN, ADJ, VERB, ADV})):
        self.embedding_reducer = embedding_reducer
        self.target_pos_set = target_pos_set

        nlp = spacy.load("en_core_web_md")
        config = {
            "model": {
                "@architectures": "spacy-transformers.TransformerModel.v1",
                "name": "bert-base-uncased",
                "tokenizer_config": {"use_fast": True}
            }
        }
        nlp.disable_pipe('parser')
        nlp.disable_pipe('ner')
        trf = nlp.add_pipe("transformer", config=config)
        trf.initialize(lambda: iter([]), nlp=nlp)
        self.nlp = nlp

    def get_word_embeddings(self, doc: Doc) -> List[Tuple[Token, np.ndarray]]:
        if len(doc) == 0:
            return []
        embeddings = doc._.trf_data.tensors[0]
        embeddings = embeddings.reshape((embeddings.shape[0] * embeddings.shape[1], embeddings.shape[2]))

        results = []
        for token, alignments in zip(doc, doc._.trf_data.align):
            if token.pos in self.target_pos_set and alignments.data.shape[0] != 0:
                token_embedding = embeddings[alignments.data[0, 0]:alignments.data[-1, 0] + 1]
                if using_gpu:
                    token_embedding = cupy.asnumpy(token_embedding)  # move the array off the GPU
                results.append((token, np.float16(self.embedding_reducer(token_embedding))))

        return results


def classify_embedding(embedding: np.ndarray, clusters: WordCluster, metric: str = 'euclidean') -> int:
    xa = np.expand_dims(embedding, axis=0)
    xb = clusters.cluster_centers
    distances = cdist(xa, xb, metric=metric)
    return np.argmin(distances)


def sort_words_by_distance(words: List[Word], vector: np.ndarray, metric: str = 'euclidian') -> List[Word]:
    xa = np.stack([x.embedding for x in words])
    xb = np.expand_dims(vector, axis=0)
    distances = cdist(xa, xb, metric=metric)

    words_with_index = list(enumerate(words))
    return [word for _, word in sorted(words_with_index, key=lambda x: distances[x[0]])]