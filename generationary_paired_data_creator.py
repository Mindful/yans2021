import sys
from pathlib import Path
import logging

import numpy as np
from tqdm import tqdm

from data.db import DbConnection, Example
from generationary.converters.utils import Context
from spacy.parts_of_speech import NOUN, ADJ, VERB, ADV
from scipy.spatial.distance import cdist


sent_separator = ' </s> '

POS_DICT = {
    'NOUN': NOUN,
    'VERB': VERB,
    'ADJ': ADJ,
    'ADV': ADV
}


def target_line_with_clusters(sentence_db: DbConnection, word_db: DbConnection, example: Example, cont: Context) -> str:
    lemma = cont.meta['lemma']
    pos = cont.meta.get('pos', None)

    if pos:
        where_clause = f'where pos={POS_DICT[pos]} and (form={lemma} or lemma={lemma})'
    else:
        where_clause = f'where form={lemma} or lemma={lemma}'

    words = list(word_db.read_words(use_tqdm=False, where_clause=where_clause))

    word_embeddings = np.ndarray([w.embedding for w in words])
    target_embedding = np.expand_dims(example.embedding, axis=0)
    distances = cdist(target_embedding, word_embeddings)[0]
    sort_indices = np.argsort(distances)[:5]

    closest_words = []
    for idx in sort_indices:
        closest_words.append(words[idx])

    closest_sentence_ids = [w.sentence for w in closest_words]
    sentences = sentence_db.read_sentences(use_tqdm=False, where_clause=f' where id IN ({",".join(str(x) for x in closest_sentence_ids)})')
    sentence_map = {
        ident: text for ident, text in sentences
    }
    closest_words = [w._replace(sentence=sentence_map[w.sentence]) for w in closest_words]

    addition_sentences = []
    for word in closest_words:
        left = word.sentence[:word.idx]
        word_end = word.sentence.find(' ', word.idx)
        if word_end == -1:
            right = ''
        else:
            right = word.sentence[word_end+1:]

        final_sentence = left.strip() + ' <define> ' + word.form + ' </define> ' + right.strip()
        addition_sentences.append(final_sentence)

    if len(closest_words) > 0:
        return cont.line_trg() + sent_separator + sent_separator.join(addition_sentences)
    else:
        return cont.line_trg()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('--run_name')
    parser.add_argument('--excluded-lemmas', type=str, required=False, default='')
    parser.add_argument('--output')

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

    word_db = args.run_name + '_words'
    example_db = args.run_name + '_examples'
    sentence_db = args.run_name + '_sentences'
    for db_name in (word_db, example_db, sentence_db):
        assert Path(db_name+'.db').exists()

    sentence_db = DbConnection(sentence_db)
    word_db = DbConnection(word_db)
    example_db = DbConnection(example_db)

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
                line_src = " " + target_line_with_clusters(sentence_db, word_db, example, cont)
                line_trg = " " + cont.line_trg().lstrip()

                if len(line_src) >= 3 and len(line_trg) >= 3:
                    f_en.write(line_src + '\n')
                    f_gl.write(line_trg + '\n')

    logging.info(f'Removed: {removed}')
