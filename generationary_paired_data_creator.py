import sys
from pathlib import Path
import logging

from data.db import DbConnection
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


def target_line_with_clusters(db: DbConnection, cont: Context) -> str:
    lemma = cont.meta['lemma']
    pos = cont.meta.get('pos', None)

    if pos:
        where_clause = f'where pos={POS_DICT[pos]} and (form={lemma} or lemma={lemma})'
    else:
        where_clause = f'form={lemma} or lemma={lemma}'

    words = db.read_words(use_tqdm=False, where_clause=where_clause)
    #TODO: we need to get an embedding for the word, either by embedding here or reading from teh DB
    # then use CDIST to compute relative distances









if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog=__file__,
        description="""
        Helper script that merges usage examples and some target (e.g. definitions, synonyms sets, etc.).
        This is necessary in order to run fairseq's own preprocessing scripts. 
        """
    )
    parser.add_argument('--contexts', nargs='+', type=Path, required=True)
    parser.add_argument('--targets', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--excluded-lemmas', type=str, required=False, default='')
    parser.add_argument('--tag', default='define')
    parser.add_argument('--embedding_db')
    args = parser.parse_args()

    assert Path(args.embedding_db+'.db').exists()
    embedding_db = DbConnection(args.embedding_db)

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

    contexts = (c for path in args.contexts for c in read_contexts(path, tag=args.tag))
    data = None
    for path in args.targets:
        data = read_trgs(path, data)
    contexts = combine_contexts_and_trgs(contexts, data, tag=args.tag)

    contexts = list(contexts)

    logging.info(f'Contexts: {len(contexts)}')
    logging.info(f'Targets: {len(data)}')
    logging.info(repr(list(data.items())[:5]))

    output_en = Path(args.output + '.raw.en').resolve()
    output_gl = Path(args.output + '.raw.gloss').resolve()

    removed = 0

    with output_en.open('w') as f_en, output_gl.open('w') as f_gl:
        for cont in contexts:
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
