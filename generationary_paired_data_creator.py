import sys
from pathlib import Path
import logging

import numpy as np
from tqdm import tqdm

from data.db import DbConnection, Example
from generationary.converters.utils import read_contexts, read_trgs, combine_contexts_and_trgs, Context
from spacy.parts_of_speech import NOUN, ADJ, VERB, ADV
from scipy.spatial.distance import cdist


sent_separator = '</s>'

POS_DICT = {
    'NOUN': NOUN,
    'VERB': VERB,
    'ADJ': ADJ,
    'ADV': ADV
}


def target_line_with_clusters(db: DbConnection, example: Example, cont: Context) -> str:
    lemma = cont.meta['lemma']
    pos = cont.meta.get('pos', None)

    if pos:
        where_clause = f'where pos={POS_DICT[pos]} and (form={lemma} or lemma={lemma})'
    else:
        where_clause = f'where form={lemma} or lemma={lemma}'

    words = list(db.read_words(use_tqdm=False, where_clause=where_clause))

    word_embeddings = np.ndarray([w.embedding for w in words])
    target_embedding = np.expand_dims(example.embedding, axis=0)
    distances = cdist(target_embedding, word_embeddings)[0]
    sort_indices = np.argsort(distances)[:5]

    closest_words = []
    for idx in sort_indices:
        closest_words.append(words[idx])

    closest_sentence_ids = [w.sentence for w in closest_words]
    #TODO: load sentences from the database


    #TODO: use CDIST to compute relative distances


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('--example_db')
    parser.add_argument('--embedding_db')
    parser.add_argument('--excluded-lemmas', type=str, required=False, default='')

    args = parser.parse_args()

    excluded_lemmas = set()
    if args.excluded_lemmas:
        for line in Path(args.excluded_lemmas).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            lemma, *other = line.split('\t')
            lemma = lemma.strip().lower()
            if not other:
                excluded_lemmas.add((lemma, None))
            else:
                excluded_lemmas.add((lemma, other[0]))

    assert Path(args.embedding_db+'.db').exists()
    assert Path(args.example_db+'.db').exists()
    embedding_db = DbConnection(args.embedding_db)
    example_db = DbConnection(args.example_db)

    examples = example_db.read_examples()

    output_en = Path(args.output + '.raw.en').resolve()
    output_gl = Path(args.output + '.raw.gloss').resolve()

    removed = 0

    with output_en.open('w') as f_en, output_gl.open('w') as f_gl:
        for example in tqdm(examples, 'processing examples'):
            cont = Context.from_context_line(example.original_line)
            lemma = cont.meta.get('lemma').strip().lower()
            pos = cont.meta.get('pos')
            if (lemma, None) in excluded_lemmas or (lemma, pos) in excluded_lemmas:
                removed += 1
                continue
            if cont.tokens_trg:
                line_src = cont.line_src()
                line_trg = cont.line_trg()

                line_src = " " + line_src.lstrip()
                line_trg = " " + line_trg.lstrip()

                if len(line_src) >= 3 and len(line_trg) >= 3:
                    f_en.write(line_src + '\n')
                    f_gl.write(line_trg + '\n')

    logging.info(f'Removed: {removed}')
