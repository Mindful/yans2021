import argparse
import logging
from pathlib import Path

import spacy
from datasets import tqdm
from generationary.converters.utils import read_contexts_with_line, combine_contexts_and_trgs, read_trgs
from data.db import DbConnection, WriteBuffer, Example
from nlp.embedding import EmbeddingExtractor
import re

special_token_re = re.compile(r'<.*?>\s')


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    parser = argparse.ArgumentParser()

    parser.add_argument('--contexts', nargs='+', type=Path, required=True)
    parser.add_argument('--targets', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--excluded-lemmas', type=str, required=False, default='')
    parser.add_argument('--tag', default='define')
    parser.add_argument('--run', required=False, default='default_run')
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

    contexts_and_lines = (c for path in args.contexts for c in read_contexts_with_line(path, tag=args.tag))
    contexts = []
    lines = []
    for context, line in tqdm(contexts_and_lines, 'reading contexts'):
        contexts.append(context)
        lines.append(line)

    data = None
    for path in args.targets:
        data = read_trgs(path, data)
    contexts = list(tqdm(combine_contexts_and_trgs(contexts, data, tag=args.tag), 'adding target data'))

    db = DbConnection(args.run)
    write_buffer = WriteBuffer('example', db.save_examples)
    spacy.require_gpu()
    extractor = EmbeddingExtractor()

    skipped = {
        'did_not_find_word': 0,
        'spaces': 0
    }

    total = 0

    for context, line in tqdm(zip(contexts, lines), 'processing contexts'):
        total += 1
        input_form = context.meta['lemma']
        if '_' in input_form:
            skipped['spaces'] += 1
            continue

        sentence = special_token_re.sub('', context.line_src())
        doc = extractor.nlp(special_token_re.sub('', context.line_src()))

        word_gen = (Example(input_form, token.text, token.lemma_.lower(), token.pos, sentence, embedding, line,
                            context.line_trg())
                    for token, embedding in extractor.get_word_embeddings(doc))

        output_word = next((x for x in word_gen if x.form == input_form.lower()), None)
        if output_word is None:
            skipped['did_not_find_word'] += 1
            continue

        write_buffer.add(output_word)

    write_buffer.flush()
    logger.info("Finished writing")
    total_skipped = sum(x for x in skipped.values())
    logger.info(f"Found {total} total examples, skipped {total_skipped} of those")
    print(str({x: y/total for x, y in skipped.items()}))


if __name__ == '__main__':
    main()
