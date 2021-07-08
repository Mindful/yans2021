from typing import Tuple, List

import numpy as np
import spacy
from spacy.parts_of_speech import NOUN, ADJ, VERB, ADV
from spacy.tokens import Doc


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


def reduce_to_first(embeddings: np.ndarray):
    return embeddings[0]


class EmbeddingExtractor:
    def __init__(self, embedding_reducer=reduce_to_first,
                 target_pos_set=frozenset({NOUN, ADJ, VERB, ADV}),
                 use_lemmas=True):
        self.embedding_reducer = lambda x: x if x.shape[0] == 0 else embedding_reducer(x)
        self.target_pos_set = target_pos_set
        self.use_lemmas = use_lemmas

    def get_word_embeddings(self, doc: Doc) -> List[Tuple[str, str, np.ndarray]]:
        embeddings = doc._.trf_data.tensors[0].squeeze()

        return [
            (token.lemma_ if self.use_lemmas else token.lower_, token.pos_,
             self.embedding_reducer(embeddings[alignments.data[0, 0]:alignments.data[-1, 0]]))
            for token, alignments in zip(doc, doc._.trf_data.align)
            if token.pos in self.target_pos_set
        ]


